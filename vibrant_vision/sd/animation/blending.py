import os
import time
from asyncio.log import logger
from itertools import zip_longest
from re import L
from typing import List, Optional, Union

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from click import prompt
from PIL import Image
from tqdm.auto import tqdm

import vibrant_vision.sd.animation.py3d_tools as p3d
from vibrant_vision.sd import constants
from vibrant_vision.sd.animation.movie import MovieSaver
from vibrant_vision.sd.models.controlnet.controlnet_hinter import hint_canny

device = constants.device
dtype = torch.float16
TRANSLATION_SCALE = 1.0 / 200.0


class LatentBlending:
    def __init__(
        self,
        sdh=None,
        upscaler=None,
        guidance_scale: float = 4.0,
        guidance_scale_mid_damper: float = 0.5,
        mid_compression_scaler: float = 1.2,
    ) -> None:
        self.sdh = sdh
        self.upscaler = upscaler
        self.width = self.sdh.width
        self.height = self.sdh.height
        self.guidance_scale_mid_damper = guidance_scale_mid_damper
        self.mid_compression_scaler = mid_compression_scaler
        self.fixed_seed = 0
        self.variance_threshold = 0.1

        # Initialize vars
        self.prompt1 = ""
        self.prompt2 = ""
        self.negative_prompt = ""

        self.tree_latents = [None, None]
        self.tree_fracts = None
        self.idx_injection = []
        self.tree_status = None
        self.tree_final_imgs = []

        self.list_nmb_branches_prev = []
        self.list_injection_idx_prev = []
        self.text_embedding1 = None
        self.text_embedding2 = None
        self.negative_embedding = None
        self.image1_lowres = None
        self.image2_lowres = None
        self.negative_prompt = None
        self.num_inference_steps = self.sdh.num_inference_steps
        self.noise_level_upscaling = 20
        self.list_injection_idx = None
        self.list_nmb_branches = None

        # Mixing parameters
        self.branch1_crossfeed_power = 0.1
        self.branch1_crossfeed_range = 0.6
        self.branch1_crossfeed_decay = 0.8

        self.parental_crossfeed_power = 0.1
        self.parental_crossfeed_range = 0.8
        self.parental_crossfeed_power_decay = 0.8

        self.set_guidance_scale(guidance_scale)
        self.init_mode()
        self.multi_transition_img_first = None
        self.multi_transition_img_last = None
        self.dt_per_diff = 0
        self.spatial_mask = None

        self.control_hint_1 = None
        self.control_hint_2 = None
        self.controlnet_guidance_percent = 0.5

        self.lpips = lpips.LPIPS(net="alex").cuda(device)

    def init_mode(self):
        self.mode = "standard"

    def set_guidance_scale(self, guidance_scale):
        r"""
        sets the guidance scale.
        """
        self.guidance_scale_base = guidance_scale
        self.guidance_scale = guidance_scale
        self.sdh.guidance_scale = guidance_scale

    def set_negative_prompt(self, negative_prompt):
        r"""Set the negative prompt. Currenty only one negative prompt is supported"""
        self.negative_prompt = negative_prompt

    def set_controlnet_guidance_percent(self, controlnet_guidance_percent):
        r"""Set the controlnet guidance percent. This is the amount of controlnet guidance that is used."""
        self.controlnet_guidance_percent = controlnet_guidance_percent

    def __set_control_image(self, image):
        r"""Set the control net latents. Given an image, the canny will be detected and used to nudge latents"""
        if isinstance(image, str):
            image = cv2.imread(image)[:, :]

        cv2.imwrite("control_image.png", image)
        control_hint = hint_canny(image, self.sdh.width, self.sdh.height)
        expected_shape = (self.height, self.width, 3)
        if control_hint.shape != expected_shape:
            raise ValueError(f"Expected image shape {expected_shape}, got {control_hint.shape}")
        cv2.imwrite("control_hint.png", control_hint)
        control_hint = torch.from_numpy(control_hint.copy())
        control_hint = control_hint.to(device=device, dtype=dtype)
        control_hint /= 255.0
        control_hint = control_hint.repeat(1, 1, 1, 1)
        control_hint = control_hint.permute(0, 3, 1, 2)
        return control_hint

    def set_control_image_1(self, image):
        self.control_hint_1 = self.__set_control_image(image)

    def set_control_image_2(self, image):
        self.control_hint_2 = self.__set_control_image(image)

    def set_guidance_mid_dampening(self, fract_mixing):
        r"""
        Tunes the guidance scale down as a linear function of fract_mixing,
        towards 0.5 the minimum will be reached.
        """
        mid_factor = 1 - np.abs(fract_mixing - 0.5) / 0.5
        max_guidance_reduction = self.guidance_scale_base * (1 - self.guidance_scale_mid_damper) - 1
        guidance_scale_effective = self.guidance_scale_base - max_guidance_reduction * mid_factor
        self.guidance_scale = guidance_scale_effective
        self.sdh.guidance_scale = guidance_scale_effective

    def set_branch1_crossfeed(self, crossfeed_power, crossfeed_range, crossfeed_decay):
        r"""
        Sets the crossfeed parameters for the first branch to the last branch.
        Args:
            crossfeed_power: float [0,1]
                Controls the level of cross-feeding between the first and last image branch.
            crossfeed_range: float [0,1]
                Sets the duration of active crossfeed during development.
            crossfeed_decay: float [0,1]
                Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
        """
        self.branch1_crossfeed_power = np.clip(crossfeed_power, 0, 1)
        self.branch1_crossfeed_range = np.clip(crossfeed_range, 0, 1)
        self.branch1_crossfeed_decay = np.clip(crossfeed_decay, 0, 1)

    def set_parental_crossfeed(self, crossfeed_power, crossfeed_range, crossfeed_decay):
        r"""
        Sets the crossfeed parameters for all transition images (within the first and last branch).
        Args:
            crossfeed_power: float [0,1]
                Controls the level of cross-feeding from the parental branches
            crossfeed_range: float [0,1]
                Sets the duration of active crossfeed during development.
            crossfeed_decay: float [0,1]
                Sets decay for branch1_crossfeed_power. Lower values make the decay stronger across the range.
        """
        self.parental_crossfeed_power = np.clip(crossfeed_power, 0, 1)
        self.parental_crossfeed_range = np.clip(crossfeed_range, 0, 1)
        self.parental_crossfeed_power_decay = np.clip(crossfeed_decay, 0, 1)

    def set_prompt1(self, prompt: str):
        r"""
        Sets the first prompt (for the first keyframe) including text embeddings.
        Args:
            prompt: str
                ABC trending on artstation painted by Greg Rutkowski
        """
        prompt = prompt.replace("_", " ")
        self.prompt1 = prompt
        self.text_embedding1 = self.get_text_embeddings(self.prompt1)

    def set_prompt2(self, prompt: str):
        r"""
        Sets the second prompt (for the second keyframe) including text embeddings.
        Args:
            prompt: str
                XYZ trending on artstation painted by Greg Rutkowski
        """
        prompt = prompt.replace("_", " ")
        self.prompt2 = prompt
        self.text_embedding2 = self.get_text_embeddings(self.prompt2)

    def set_image1(self, image: Image):
        r"""
        Sets the first image (keyframe), relevant for the upscaling model transitions.
        Args:
            image: Image
        """
        self.image1_lowres = image

    def set_image2(self, image: Image):
        r"""
        Sets the second image (keyframe), relevant for the upscaling model transitions.
        Args:
            image: Image
        """
        self.image2_lowres = image

    def run_transition(
        self,
        recycle_img1: Optional[bool] = False,
        recycle_img2: Optional[bool] = False,
        num_inference_steps: Optional[int] = 30,
        depth_strength: Optional[float] = 0.3,
        t_compute_max_allowed: Optional[float] = None,
        nmb_max_branches: Optional[int] = None,
        fixed_seed: Optional[int] = None,
        variance_threshold: Optional[float] = 0.1,
        max_frames: Optional[int] = None,
    ):
        r"""
        Function for computing transitions.
        Returns a list of transition images using spherical latent blending.
        Args:
            recycle_img1: Optional[bool]:
                Don't recompute the latents for the first keyframe (purely prompt1). Saves compute.
            recycle_img2: Optional[bool]:
                Don't recompute the latents for the second keyframe (purely prompt2). Saves compute.
            num_inference_steps:
                Number of diffusion steps. Higher values will take more compute time.
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max_allowed:
                Either provide t_compute_max_allowed or nmb_max_branches.
                The maximum time allowed for computation. Higher values give better results but take longer.
            nmb_max_branches: int
                Either provide t_compute_max_allowed or nmb_max_branches. The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent
                of your computer.
            fixed_seeds: Optional[List[int)]:
                You can supply two seeds that are used for the first and second keyframe (prompt1 and prompt2).
                Otherwise random seeds will be taken.
            max_frames: Optional[int]:
                The maximum number of frames to be computed
        """

        # Sanity checks first
        assert self.text_embedding1 is not None, "Set the first text embedding with .set_prompt1(...) before"
        assert self.text_embedding2 is not None, "Set the second text embedding with .set_prompt2(...) before"

        # Random seeds
        if fixed_seed is not None:
            self.fixed_seed = fixed_seed
        else:
            self.fixed_seed = np.random.randint(0, 2**31 - 1).astype(np.int32)

        self.variance_threshold = variance_threshold

        # Ensure correct num_inference_steps in holder
        self.num_inference_steps = num_inference_steps
        self.sdh.num_inference_steps = num_inference_steps

        # Compute / Recycle first image
        if not recycle_img1 or len(self.tree_latents[0]) != self.num_inference_steps:
            list_latents1 = self.compute_latents1()
            img1 = self.sdh.decode_latents(list_latents1[-1])
            self.set_control_image_1(img1)
        else:
            list_latents1 = self.tree_latents[0]

        # Compute / Recycle second image
        if not recycle_img2 or len(self.tree_latents[-1]) != self.num_inference_steps:
            list_latents2 = self.compute_latents2()
            img2 = self.sdh.decode_latents(list_latents2[-1])
            self.set_control_image_2(img2)
        else:
            list_latents2 = self.tree_latents[-1]

        # Reset the tree, injecting the edge latents1/2 we just generated/recycled
        self.tree_latents = [list_latents1, list_latents2]
        self.tree_fracts = [0.0, 1.0]
        self.tree_final_imgs = [
            self.sdh.decode_latents((self.tree_latents[0][-1])),
            self.sdh.decode_latents((self.tree_latents[-1][-1])),
        ]
        self.tree_idx_injection = [0, 0]

        # Hard-fix. Apply spatial mask only for list_latents2 but not for transition. WIP...
        self.spatial_mask = None

        # Set up branching scheme (dependent on provided compute time)
        list_idx_injection, list_nmb_stems = self.get_time_based_branching(
            depth_strength, t_compute_max_allowed, nmb_max_branches, max_frames
        )

        # Run iteratively, starting with the longest trajectory.
        # Always inserting new branches where they are needed most according to image similarity
        for s_idx in tqdm(range(len(list_idx_injection))):
            nmb_stems = list_nmb_stems[s_idx]
            idx_injection = list_idx_injection[s_idx]

            logger.info(f"Starting iteration {s_idx} with {nmb_stems} stems and idx_injection {idx_injection}")

            for i in range(nmb_stems):
                fract_mixing, b_parent1, b_parent2 = self.get_mixing_parameters(idx_injection)
                self.set_guidance_mid_dampening(fract_mixing)
                list_latents = self.compute_latents_mix(fract_mixing, b_parent1, b_parent2, idx_injection)
                self.insert_into_tree(fract_mixing, idx_injection, list_latents)

        self.upscale_decode_latents()
        return self.tree_final_imgs

    def compute_latents1(self, return_image=False):
        r"""
        Runs a diffusion trajectory for the first image
        Args:
            return_image: bool
                whether to return an image or the list of latents
        """
        logger.info("Computing latents for first image")
        list_conditionings = self.get_mixed_conditioning(0)
        t0 = time.time()
        latents_start = self.get_noise()
        latents_start = latents_start * self.sdh.scheduler.init_noise_sigma
        list_latents1 = self.run_diffusion(list_conditionings, latents_start=latents_start, idx_start=0)
        t1 = time.time()
        self.dt_per_diff = (t1 - t0) / self.num_inference_steps
        self.tree_latents[0] = list_latents1
        if return_image:
            return self.sdh.decode_latents(list_latents1[-1])
        else:
            return list_latents1

    def compute_latents2(self, return_image=False):
        r"""
        Runs a diffusion trajectory for the last image, which may be affected by the first image's trajectory.
        Args:
            return_image: bool
                whether to return an image or the list of latents
        """
        logger.info("Computing latents for second image")
        list_conditionings = self.get_mixed_conditioning(1)
        latents_start = self.get_noise()
        # Influence from branch1
        if self.branch1_crossfeed_power > 0.0:
            # Set up the mixing_coeffs
            idx_mixing_stop = int(round(self.num_inference_steps * self.branch1_crossfeed_range))
            mixing_coeffs = list(
                np.linspace(
                    self.branch1_crossfeed_power,
                    self.branch1_crossfeed_power * self.branch1_crossfeed_decay,
                    idx_mixing_stop,
                )
            )
            mixing_coeffs.extend((self.num_inference_steps - idx_mixing_stop) * [0])
            list_latents_mixing = self.tree_latents[0]
            list_latents2 = self.run_diffusion(
                list_conditionings,
                latents_start=latents_start,
                idx_start=0,
                list_latents_mixing=list_latents_mixing,
                mixing_coeffs=mixing_coeffs,
            )
        else:
            list_latents2 = self.run_diffusion(list_conditionings, latents_start)
        self.tree_latents[-1] = list_latents2

        if return_image:
            return self.sdh.decode_latents(list_latents2[-1])
        else:
            return list_latents2

    def compute_latents_mix(self, fract_mixing, b_parent1, b_parent2, idx_injection):
        r"""
        Runs a diffusion trajectory, using the latents from the respective parents
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            b_parent1: int
                index of parent1 to be used
            b_parent2: int
                index of parent2 to be used
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        list_conditionings = self.get_mixed_conditioning(fract_mixing)
        control_hint = self.get_mixed_control_hint(fract_mixing)

        fract_mixing_parental = (fract_mixing - self.tree_fracts[b_parent1]) / (
            self.tree_fracts[b_parent2] - self.tree_fracts[b_parent1]
        )
        # idx_reversed = self.num_inference_steps - idx_injection

        list_latents_parental_mix = []
        for i in range(self.num_inference_steps):
            latents_p1 = self.tree_latents[b_parent1][i]
            latents_p2 = self.tree_latents[b_parent2][i]
            if latents_p1 is None or latents_p2 is None:
                latents_parental = None
            else:
                latents_parental = interpolate_spherical(latents_p1, latents_p2, fract_mixing_parental)
            list_latents_parental_mix.append(latents_parental)

        idx_mixing_stop = int(round(self.num_inference_steps * self.parental_crossfeed_range))
        mixing_coeffs = idx_injection * [self.parental_crossfeed_power]
        nmb_mixing = idx_mixing_stop - idx_injection
        if nmb_mixing > 0:
            mixing_coeffs.extend(
                list(
                    np.linspace(
                        self.parental_crossfeed_power,
                        self.parental_crossfeed_power * self.parental_crossfeed_power_decay,
                        nmb_mixing,
                    )
                )
            )
        mixing_coeffs.extend((self.num_inference_steps - len(mixing_coeffs)) * [0])

        latents_start = list_latents_parental_mix[idx_injection - 1]
        list_latents = self.run_diffusion(
            list_conditionings,
            latents_start=latents_start,
            idx_start=idx_injection,
            list_latents_mixing=list_latents_parental_mix,
            mixing_coeffs=mixing_coeffs,
            controlnet_hint=control_hint,
        )

        return list_latents

    def get_time_based_branching(
        self, depth_strength, t_compute_max_allowed=None, nmb_max_branches=None, max_frames=None
    ):
        r"""
        Sets up the branching scheme dependent on the time that is granted for compute.
        The scheme uses an estimation derived from the first image's computation speed.
        Either provide t_compute_max_allowed or nmb_max_branches
        Args:
            depth_strength:
                Determines how deep the first injection will happen.
                Deeper injections will cause (unwanted) formation of new structures,
                more shallow values will go into alpha-blendy land.
            t_compute_max_allowed: float
                The maximum time allowed for computation. Higher values give better results
                but take longer. Use this if you want to fix your waiting time for the results.
            nmb_max_branches: int
                The maximum number of branches to be computed. Higher values give better
                results. Use this if you want to have controllable results independent
                of your computer.
        """
        idx_injection_base = int(round(self.num_inference_steps * depth_strength))
        list_idx_injection = np.arange(idx_injection_base, self.num_inference_steps - 1, 3)
        list_nmb_stems = np.ones(len(list_idx_injection), dtype=np.int32)
        t_compute = 0

        if max_frames is not None:
            stop_criterion = "max_frames"
        elif nmb_max_branches is None:
            assert t_compute_max_allowed is not None, "Either specify t_compute_max_allowed or nmb_max_branches"
            stop_criterion = "t_compute_max_allowed"
        elif t_compute_max_allowed is None:
            assert nmb_max_branches is not None, "Either specify t_compute_max_allowed or nmb_max_branches"
            stop_criterion = "nmb_max_branches"
            nmb_max_branches -= 2  # discounting the outer frames
        else:
            raise ValueError("Either specify t_compute_max_allowed or nmb_max_branches")

        stop_criterion_reached = False
        is_first_iteration = True

        while not stop_criterion_reached:
            list_compute_steps = self.num_inference_steps - list_idx_injection
            list_compute_steps *= list_nmb_stems
            t_compute = np.sum(list_compute_steps) * self.dt_per_diff + 0.15 * np.sum(list_nmb_stems)
            increase_done = False
            for s_idx in range(len(list_nmb_stems) - 1):
                if list_nmb_stems[s_idx + 1] / list_nmb_stems[s_idx] >= 2:
                    list_nmb_stems[s_idx] += 1
                    increase_done = True
                    break
            if not increase_done:
                list_nmb_stems[-1] += 1

            if stop_criterion == "t_compute_max_allowed" and t_compute > t_compute_max_allowed:
                stop_criterion_reached = True
            elif stop_criterion == "nmb_max_branches" and np.sum(list_nmb_stems) >= nmb_max_branches:
                stop_criterion_reached = True
                if is_first_iteration:
                    # Need to undersample.
                    list_idx_injection = np.linspace(
                        list_idx_injection[0], list_idx_injection[-1], nmb_max_branches
                    ).astype(np.int32)
                    list_nmb_stems = np.ones(len(list_idx_injection), dtype=np.int32)
            elif stop_criterion == "max_frames" and np.sum(list_nmb_stems) + 3 > max_frames:
                stop_criterion_reached = True
            else:
                is_first_iteration = False

        return list_idx_injection, list_nmb_stems

    def get_mixing_parameters(self, idx_injection):
        r"""
        Computes which parental latents should be mixed together to achieve a smooth blend.
        As metric, we are using lpips image similarity. The insertion takes place
        where the metric is maximal.
        Args:
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
        """
        # get_lpips_similarity
        similarities = []
        for i in range(len(self.tree_final_imgs) - 1):
            similarities.append(self.get_lpips_similarity(self.tree_final_imgs[i], self.tree_final_imgs[i + 1]))
        b_closest1 = np.argmax(similarities)
        b_closest2 = b_closest1 + 1
        fract_closest1 = self.tree_fracts[b_closest1]
        fract_closest2 = self.tree_fracts[b_closest2]

        # Ensure that the parents are indeed older!
        b_parent1 = b_closest1
        while True:
            if self.tree_idx_injection[b_parent1] < idx_injection:
                break
            else:
                b_parent1 -= 1

        b_parent2 = b_closest2
        while True:
            if self.tree_idx_injection[b_parent2] < idx_injection:
                break
            else:
                b_parent2 += 1

        # print(f"\n\nb_closest: {b_closest1} {b_closest2} fract_closest1 {fract_closest1} fract_closest2 {fract_closest2}")
        # print(f"b_parent: {b_parent1} {b_parent2}")
        # print(f"similarities {similarities}")
        # print(f"idx_injection {idx_injection} tree_idx_injection {self.tree_idx_injection}")

        fract_mixing = (fract_closest1 + fract_closest2) / 2
        return fract_mixing, b_parent1, b_parent2

    def insert_into_tree(self, fract_mixing, idx_injection, list_latents):
        r"""
        Inserts all necessary parameters into the trajectory tree.
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
            idx_injection: int
                the index in terms of diffusion steps, where the next insertion will start.
            list_latents: list
                list of the latents to be inserted
        """
        b_parent1, b_parent2 = get_closest_idx(fract_mixing, self.tree_fracts)
        self.tree_latents.insert(b_parent1 + 1, list_latents)
        self.tree_final_imgs.insert(b_parent1 + 1, self.sdh.decode_latents(list_latents[-1]))
        self.tree_fracts.insert(b_parent1 + 1, fract_mixing)
        self.tree_idx_injection.insert(b_parent1 + 1, idx_injection)

    def upscale_decode_latents(
        self,
    ):
        if self.upscaler is None:
            return

        batch_size = 4
        result = []

        def grouper(iterable, n, fillvalue=None):
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)

        for latents in grouper(self.tree_latents, batch_size):
            latents = [latent[-1] for latent in latents if latent is not None]
            latents = torch.cat(latents, dim=0)
            images = self.upscaler(
                latents=latents,
                num_inference_steps=self.num_inference_steps,
                return_image=True,
                batch_size=min(batch_size, len(latents)),
            )
            result.extend(images)

        self.tree_final_imgs = result

    def get_spatial_mask_template(self):
        r"""
        Experimental helper function to get a spatial mask template.
        """
        shape_latents = [self.sdh.C, self.sdh.height // self.sdh.f, self.sdh.width // self.sdh.f]
        C, H, W = shape_latents
        return np.ones((H, W))

    def set_spatial_mask(self, img_mask):
        r"""
        Experimental helper function to set a spatial mask.
        The mask forces latents to be overwritten.
        Args:
            img_mask:
                mask image [0,1]. You can get a template using get_spatial_mask_template

        """

        shape_latents = [self.sdh.C, self.sdh.height // self.sdh.f, self.sdh.width // self.sdh.f]
        C, H, W = shape_latents
        img_mask = np.asarray(img_mask)
        assert len(img_mask.shape) == 2, "Currently, only 2D images are supported as mask"
        img_mask = np.clip(img_mask, 0, 1)
        assert img_mask.shape[0] == H, f"Your mask needs to be of dimension {H} x {W}"
        assert img_mask.shape[1] == W, f"Your mask needs to be of dimension {H} x {W}"
        spatial_mask = torch.from_numpy(img_mask).to(device=self.device)
        spatial_mask = torch.unsqueeze(spatial_mask, 0)
        spatial_mask = spatial_mask.repeat((C, 1, 1))
        spatial_mask = torch.unsqueeze(spatial_mask, 0)

        self.spatial_mask = spatial_mask

    def get_noise(self):
        r"""
        Helper function to get noise given seed.
        Args:
            seed: int

        """

        if self.mode == "standard":
            shape_latents = [
                self.sdh.channels,
                self.sdh.height // self.sdh.down_sample_factor,
                self.sdh.width // self.sdh.down_sample_factor,
            ]
            C, H, W = shape_latents
        elif self.mode == "upscale":
            w = self.image1_lowres.size[0]
            h = self.image1_lowres.size[1]
            shape_latents = [self.sdh.channels, h, w]
            C, H, W = shape_latents

        generator = torch.Generator(device=device).manual_seed(np.random.randint(0, 2**31 - 1))
        subnoise = torch.randn((1, C, H, W), generator=generator, device=device, dtype=dtype)

        generator = torch.Generator(device=device).manual_seed(int(self.fixed_seed))
        noise = torch.randn((1, C, H, W), generator=generator, device=device, dtype=dtype)

        if subnoise is not None:
            noise = slerp(noise, subnoise, self.variance_threshold)

        return noise

    @torch.no_grad()
    def run_diffusion(
        self,
        list_conditionings,
        controlnet_hint=None,
        latents_start: torch.FloatTensor = None,
        idx_start: int = 0,
        list_latents_mixing=None,
        mixing_coeffs=0.0,
        return_image: Optional[bool] = False,
    ):

        r"""
        Wrapper function for diffusion runners.
        Depending on the mode, the correct one will be executed.

        Args:
            list_conditionings: list
                List of all conditionings for the diffusion model.
            latents_start: torch.FloatTensor
                Latents that are used for injection
            idx_start: int
                Index of the diffusion process start and where the latents_for_injection are injected
            list_latents_mixing: torch.FloatTensor
                List of latents (latent trajectories) that are used for mixing
            mixing_coeffs: float or list
                Coefficients, how strong each element of list_latents_mixing will be mixed in.
            return_image: Optional[bool]
                Optionally return image directly
        """

        # Ensure correct num_inference_steps in Holder
        self.sdh.num_inference_steps = self.num_inference_steps
        assert type(list_conditionings) is list, "list_conditionings need to be a list"

        if self.mode == "standard":
            text_embeddings = list_conditionings[0]
            return self.sdh(
                prompt_embeds=text_embeddings,
                latents=latents_start,
                idx_start=idx_start,
                list_latents_mixing=list_latents_mixing,
                mixing_coeffs=mixing_coeffs,
                spatial_mask=self.spatial_mask,
                controlnet_hint=controlnet_hint,
                return_image=return_image,
                controlnet_guidance_percent=self.controlnet_guidance_percent,
            )

    def get_wrap_offset(self):
        trans_xyz = [-0, 0, -0 * TRANSLATION_SCALE]
        rot_xyz = [0, 0, 0]

        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rot_xyz, device=device), "XYZ").unsqueeze(0)
        w, h = self.sdh.width // self.sdh.down_sample_factor, self.sdh.height // self.sdh.down_sample_factor

        aspect_ratio = float(w) / float(h)
        near_plane, far_plane, fov_deg = 200, 10000, 40
        persp_cam_old = p3d.FoVPerspectiveCameras(
            near_plane, far_plane, aspect_ratio, fov=fov_deg, degrees=True, device=device
        )
        persp_cam_new = p3d.FoVPerspectiveCameras(
            near_plane,
            far_plane,
            aspect_ratio,
            fov=fov_deg,
            degrees=True,
            R=rot_mat,
            T=torch.tensor([trans_xyz]),
            device=device,
        )
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=device),
            torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=device),
        )

        z = torch.ones_like(x)

        xyz_world_old = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
        xyz_cam_xy_old = persp_cam_old.get_full_projection_transform().transform_points(xyz_world_old)[:, 0:2]
        xyz_cam_xy_new = persp_cam_new.get_full_projection_transform().transform_points(xyz_world_old)[:, 0:2]

        offset_xy = xyz_cam_xy_new - xyz_cam_xy_old
        identity_2d_batch = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0)
        coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
        offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

        return offset_coords_2d.half()

    @torch.no_grad()
    def get_mixed_conditioning(self, fract_mixing):
        if self.mode == "standard":
            text_embeddings_mix = interpolate_linear(self.text_embedding1, self.text_embedding2, fract_mixing)
            list_conditionings = [text_embeddings_mix]
        else:
            raise ValueError(f"mix_conditioning: unknown mode {self.mode}")
        return list_conditionings

    @torch.no_grad()
    def get_mixed_control_hint(self, fract_mixing):
        s = interpolate_linear(self.control_hint_1, self.control_hint_2, fract_mixing)
        return s

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str):
        r"""
        Computes the text embeddings provided a string with a prompts.
        Adapted from stable diffusion repo
        Args:
            prompt: str
                ABC trending on artstation painted by Old Greg.
        """

        return self.sdh.get_text_embeds(prompt=prompt, negative_prompt=self.negative_prompt)

    def write_imgs_transition(self, dp_img):
        r"""
        Writes the transition images into the folder dp_img.
        Requires run_transition to be completed.
        Args:
            dp_img: str
                Directory, into which the transition images, yaml file and latents are written.
        """
        imgs_transition = self.tree_final_imgs
        os.makedirs(dp_img, exist_ok=True)
        for i, img in enumerate(imgs_transition):
            img_leaf = Image.fromarray(img)
            img_leaf.save(os.path.join(dp_img, f"lowres_img_{str(i).zfill(4)}.jpg"))

        fp_yml = os.path.join(dp_img, "lowres.yaml")
        self.save_statedict(fp_yml)

    def write_movie_transition(self, fp_movie, duration_transition, fps=30):
        r"""
        Writes the transition movie to fp_movie, using the given duration and fps..
        The missing frames are linearly interpolated.
        Args:
            fp_movie: str
                file pointer to the final movie.
            duration_transition: float
                duration of the movie in seonds
            fps: int
                fps of the movie

        """

        # Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
        imgs_transition_ext = add_frames_linear_interp(self.tree_final_imgs, duration_transition, fps)
        h, w = imgs_transition_ext[0].shape[:2]

        # Save as MP4
        if os.path.isfile(fp_movie):
            os.remove(fp_movie)
        ms = MovieSaver(fp_movie, fps=fps, shape_hw=[h, w])
        for img in tqdm(imgs_transition_ext):
            ms.write_frame(img)
        ms.finalize()

    def save_statedict(self, fp_yml):
        # Dump everything relevant into yaml
        imgs_transition = self.tree_final_imgs
        state_dict = self.get_state_dict()
        state_dict["nmb_images"] = len(imgs_transition)
        yml_save(fp_yml, state_dict)

    def get_state_dict(self):
        state_dict = {}
        grab_vars = [
            "prompt1",
            "prompt2",
            "seed1",
            "seed2",
            "height",
            "width",
            "num_inference_steps",
            "depth_strength",
            "guidance_scale",
            "guidance_scale_mid_damper",
            "mid_compression_scaler",
            "negative_prompt",
            "branch1_crossfeed_power",
            "branch1_crossfeed_range",
            "branch1_crossfeed_decay" "parental_crossfeed_power",
            "parental_crossfeed_range",
            "parental_crossfeed_power_decay",
        ]
        for v in grab_vars:
            if hasattr(self, v):
                if v == "seed1" or v == "seed2":
                    state_dict[v] = int(getattr(self, v))
                elif v == "guidance_scale":
                    state_dict[v] = float(getattr(self, v))

                else:
                    try:
                        state_dict[v] = getattr(self, v)
                    except Exception as e:
                        pass

        return state_dict

    def randomize_seed(self):
        r"""
        Set a random seed for a fresh start.
        """
        seed = np.random.randint(0, 2**31 - 1)
        self.set_seed(seed)

    def set_seed(self, seed: int):
        r"""
        Set a the seed for a fresh start.
        """
        self.seed = seed
        self.sdh.seed = seed

    def set_width(self, width):
        r"""
        Set the width of the resulting image.
        """
        assert np.mod(width, 64) == 0, "set_width: value needs to be divisible by 64"
        self.width = width
        self.sdh.width = width

    def set_height(self, height):
        r"""
        Set the height of the resulting image.
        """
        assert np.mod(height, 64) == 0, "set_height: value needs to be divisible by 64"
        self.height = height
        self.sdh.height = height

    def swap_forward(self):
        r"""
        Moves over keyframe two -> keyframe one. Useful for making a sequence of transitions
        as in run_multi_transition()
        """
        # Move over all latents
        self.tree_latents[0] = self.tree_latents[-1]

        # Move over prompts and text embeddings
        self.prompt1 = self.prompt2
        self.text_embedding1 = self.text_embedding2

        # Move over control hints
        self.control_hint_1 = self.control_hint_2

        # Final cleanup for extra sanity
        self.tree_final_imgs = []

    def get_lpips_similarity(self, imgA, imgB):
        r"""
        Computes the image similarity between two images imgA and imgB.
        Used to determine the optimal point of insertion to create smooth transitions.
        High values indicate low similarity.
        """
        tensorA = torch.from_numpy(imgA).float().cuda(device)
        tensorA = 2 * tensorA / 255.0 - 1
        tensorA = tensorA.permute([2, 0, 1]).unsqueeze(0)

        tensorB = torch.from_numpy(imgB).float().cuda(device)
        tensorB = 2 * tensorB / 255.0 - 1
        tensorB = tensorB.permute([2, 0, 1]).unsqueeze(0)
        lploss = self.lpips(tensorA, tensorB)
        lploss = float(lploss[0][0][0][0])

        return lploss


# Auxiliary functions
def get_closest_idx(
    fract_mixing: float,
    list_fract_mixing_prev: List[float],
):
    r"""
    Helper function to retrieve the parents for any given mixing.
    Example: fract_mixing = 0.4 and list_fract_mixing_prev = [0, 0.3, 0.6, 1.0]
    Will return the two closest values from list_fract_mixing_prev, i.e. [1, 2]
    """

    pdist = fract_mixing - np.asarray(list_fract_mixing_prev)
    pdist_pos = pdist.copy()
    pdist_pos[pdist_pos < 0] = np.inf
    b_parent1 = np.argmin(pdist_pos)
    pdist_neg = -pdist.copy()
    pdist_neg[pdist_neg <= 0] = np.inf
    b_parent2 = np.argmin(pdist_neg)

    if b_parent1 > b_parent2:
        tmp = b_parent2
        b_parent2 = b_parent1
        b_parent1 = tmp

    return b_parent1, b_parent2


@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r"""
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Slerp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0:
            First tensor for interpolation
        p1:
            Second tensor for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """

    if p0.dtype == torch.float16:
        recast_to = "fp16"
    else:
        recast_to = "fp32"

    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == "fp16":
        interp = interp.half()
    elif recast_to == "fp32":
        interp = interp.float()

    return interp


def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0:
            First tensor / np.ndarray for interpolation
        p1:
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == "uint8":
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)

    if type(p1) is np.ndarray and p1.dtype == "uint8":
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)

    # Pad the shorter one to the longer one
    if p0.shape[1] != p1.shape[1]:
        if p0.shape[1] > p1.shape[1]:
            p1 = F.pad(p1, (0, 0, 0, p0.shape[1] - p1.shape[1]))
        else:
            p0 = F.pad(p0, (0, 0, 0, p1.shape[1] - p0.shape[1]))

    interp = (1 - fract_mixing) * p0 + fract_mixing * p1

    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)

    return interp


def add_frames_linear_interp(
    list_imgs: List[np.ndarray],
    fps_target: Union[float, int] = None,
    duration_target: Union[float, int] = None,
    nmb_frames_target: int = None,
):
    r"""
    Helper function to cheaply increase the number of frames given a list of images,
    by virtue of standard linear interpolation.
    The number of inserted frames will be automatically adjusted so that the total of number
    of frames can be fixed precisely, using a random shuffling technique.
    The function allows 1:1 comparisons between transitions as videos.

    Args:
        list_imgs: List[np.ndarray)
            List of images, between each image new frames will be inserted via linear interpolation.
        fps_target:
            OptionA: specify here the desired frames per second.
        duration_target:
            OptionA: specify here the desired duration of the transition in seconds.
        nmb_frames_target:
            OptionB: directly fix the total number of frames of the output.
    """

    # Sanity
    if nmb_frames_target is not None and fps_target is not None:
        raise ValueError("You cannot specify both fps_target and nmb_frames_target")
    if fps_target is None:
        assert nmb_frames_target is not None, "Either specify nmb_frames_target or nmb_frames_target"
    if nmb_frames_target is None:
        assert fps_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        assert duration_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        nmb_frames_target = fps_target * duration_target

    # Get number of frames that are missing
    nmb_frames_diff = len(list_imgs) - 1
    nmb_frames_missing = nmb_frames_target - nmb_frames_diff - 1

    if nmb_frames_missing < 1:
        return list_imgs

    list_imgs_float = [img.astype(np.float32) for img in list_imgs]
    # Distribute missing frames, append nmb_frames_to_insert(i) frames for each frame
    mean_nmb_frames_insert = nmb_frames_missing / nmb_frames_diff
    constfact = np.floor(mean_nmb_frames_insert)
    remainder_x = 1 - (mean_nmb_frames_insert - constfact)

    nmb_iter = 0
    while True:
        nmb_frames_to_insert = np.random.rand(nmb_frames_diff)
        nmb_frames_to_insert[nmb_frames_to_insert <= remainder_x] = 0
        nmb_frames_to_insert[nmb_frames_to_insert > remainder_x] = 1
        nmb_frames_to_insert += constfact
        if np.sum(nmb_frames_to_insert) == nmb_frames_missing:
            break
        nmb_iter += 1
        if nmb_iter > 100000:
            print("add_frames_linear_interp: issue with inserting the right number of frames")
            break

    nmb_frames_to_insert = nmb_frames_to_insert.astype(np.int32)
    list_imgs_interp = []
    with tqdm(total=len(list_imgs_float) - 1, desc="STAGE linear interp") as pbar:
        for i in range(len(list_imgs_float) - 1):  # , desc="STAGE linear interp"):
            img0 = list_imgs_float[i]
            img1 = list_imgs_float[i + 1]
            list_imgs_interp.append(img0.astype(np.uint8))
            list_fracts_linblend = np.linspace(0, 1, nmb_frames_to_insert[i] + 2)[1:-1]
            output = rife_interpolate(img0, img1, len(list_fracts_linblend))
            list_imgs_interp.extend(output)
            pbar.update()
            # for fract_linblend in list_fracts_linblend:
            # img_blend = interpolate_linear(img0, img1, fract_linblend).astype(np.uint8)
            # list_imgs_interp.append(img_blend.astype(np.uint8))

        if i == len(list_imgs_float) - 2:
            list_imgs_interp.append(img1.astype(np.uint8))

    return list_imgs_interp


from vibrant_vision.sd.animation.rife.RIFE_HDv3 import Model as RIFE

rife_model = RIFE(arbitrary=True)
rife_model.load_model("./vibrant_vision/sd/animation/rife/flownet-v6-m.pkl")
rife_model.eval()
rife_model.device()


def rife_interpolate(img1, img2, tot_frame, scale=1.0):
    h, w, _ = img1.shape
    tmp = int(32 * scale)
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    i1 = torch.from_numpy(np.transpose(img1, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.0
    i2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))).to(device).unsqueeze(0).float() / 255.0
    i1 = F.pad(i1, padding)
    i2 = F.pad(i2, padding)

    def execute(i1, i2, n):
        with torch.no_grad():
            # if rife_model.version >= 3.9:
            # res = []
            # for i in range(n):
            #     res.append(rife_model.inference(i1, i2, timestep=(i + 1) * 1.0 / (n + 1), scale=scale))
            # return res
            # else:
            mid = rife_model.inference(i1, i2, scale=scale)
            if n == 1:
                return [mid]
            first_half = execute(i1, mid, n // 2)
            second_half = execute(mid, i2, n // 2)
            if n % 2:
                return [*first_half, mid, *second_half]
            return [*first_half, *second_half]

    output = execute(i1, i2, tot_frame)
    result = []
    for mid in output:
        mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
        mid = mid[:h, :w]
        result.append(mid)
    return result


def get_spacing(nmb_points: int, scaling: float):
    """
    Helper function for getting nonlinear spacing between 0 and 1, symmetric around 0.5
    Args:
        nmb_points: int
            Number of points between [0, 1]
        scaling: float
            Higher values will return higher sampling density around 0.5

    """
    if scaling < 1.7:
        return np.linspace(0, 1, nmb_points)
    nmb_points_per_side = nmb_points // 2 + 1
    if np.mod(nmb_points, 2) != 0:  # uneven case
        left_side = np.abs(np.linspace(1, 0, nmb_points_per_side) ** scaling / 2 - 0.5)
        right_side = 1 - left_side[::-1][1:]
    else:
        left_side = np.abs(np.linspace(1, 0, nmb_points_per_side) ** scaling / 2 - 0.5)[0:-1]
        right_side = 1 - left_side[::-1]
    all_fracts = np.hstack([left_side, right_side])
    return all_fracts


def get_time(resolution=None):
    """
    Helper function returning an nicely formatted time string, e.g. 221117_1620
    """
    if resolution == None:
        resolution = "second"
    if resolution == "day":
        t = time.strftime("%y%m%d", time.localtime())
    elif resolution == "minute":
        t = time.strftime("%y%m%d_%H%M", time.localtime())
    elif resolution == "second":
        t = time.strftime("%y%m%d_%H%M%S", time.localtime())
    elif resolution == "millisecond":
        t = time.strftime("%y%m%d_%H%M%S", time.localtime())
        t += "_"
        t += str("{:03d}".format(int(int(datetime.utcnow().strftime("%f")) / 1000)))
    else:
        raise ValueError("bad resolution provided: %s" % resolution)
    return t


def compare_dicts(a, b):
    """
    Compares two dictionaries a and b and returns a dictionary c, with all
    keys,values that have shared keys in a and b but same values in a and b.
    The values of a and b are stacked together in the output.
    Example:
        a = {}; a['bobo'] = 4
        b = {}; b['bobo'] = 5
        c = dict_compare(a,b)
        c = {"bobo",[4,5]}
    """
    c = {}
    for key in a.keys():
        if key in b.keys():
            val_a = a[key]
            val_b = b[key]
            if val_a != val_b:
                c[key] = [val_a, val_b]
    return c


def yml_load(fp_yml, print_fields=False):
    """
    Helper function for loading yaml files
    """
    with open(fp_yml) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    dict_data = dict(data)
    print("load: loaded {}".format(fp_yml))
    return dict_data


def yml_save(fp_yml, dict_stuff):
    """
    Helper function for saving yaml files
    """
    with open(fp_yml, "w") as f:
        data = yaml.dump(dict_stuff, f, sort_keys=False, default_flow_style=False)
    print("yml_save: saved {}".format(fp_yml))


def slerp(low, high, val):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)

    if dot.mean() > 0.9995:
        # linear
        return low * val + high * (1 - val)

    # spherical
    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res
