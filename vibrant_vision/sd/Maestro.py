from pathlib import Path

import numpy as np
import torch
from PIL import Image

import vibrant_vision.sd.animation.movie as movie
from vibrant_vision.general.logging import get_logger
from vibrant_vision.sd import constants
from vibrant_vision.sd.animation.blending import LatentBlending, rife_interpolate
from vibrant_vision.sd.animation.keyframes import KeyFrames
from vibrant_vision.sd.animation.wrap import FrameWrapper
from vibrant_vision.sd.models.ImageModel import BlenderPipeline
from vibrant_vision.sd.models.controlnet.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel

logger = get_logger(__name__)
device = constants.device
checkpoint = "WarriorMama777/AbyssOrangeMix2"
revision = "main"
dtype = torch.float16
controlnet_path = "./models/control_sd15_canny_fp16"


class Maestro:
    models_path = Path(__file__).parent / "models"
    configs_path = Path(__file__).parent / "configs"
    output_path = Path(__file__).parent / "output"

    def __init__(self, fps) -> None:
        self.fps = fps
        self.total_frames = 15 * self.fps  # Hardcoded duration for now: 15 seconds at 30 fps = 450 frames
        self.target_latents_frames = self.fps * 2
        self.with_target_latents = False

        logger.info("Initializing animation keyframes...")
        self.translation_x_keys = KeyFrames(self.total_frames)
        self.translation_y_keys = KeyFrames(self.total_frames)
        self.translation_z_keys = KeyFrames(self.total_frames)
        self.rotation_x_keys = KeyFrames(self.total_frames)
        self.rotation_y_keys = KeyFrames(self.total_frames)
        self.rotation_z_keys = KeyFrames(self.total_frames)

        logger.info("Initializing model keyframes...")
        self.prompt_keys = KeyFrames(self.total_frames)
        self.num_inference_steps_keys = KeyFrames(self.total_frames)
        self.seed_keys = KeyFrames(self.total_frames)
        self.strength_keys = KeyFrames(self.total_frames)
        self.noise_keys = KeyFrames(self.total_frames)
        self.target_latents_keys = KeyFrames(self.total_frames, any_type=True)

        for i in range(self.total_frames):
            self.translation_x_keys[i] = 0.0  # 5.0 * np.sin(2 * np.pi * i / self.fps)
            self.translation_y_keys[i] = 0.0
            self.translation_z_keys[i] = 0.9  # np.cos(i / self.fps * (2 / 3)) + i / self.fps * (1 / 5)
            self.rotation_x_keys[i] = 0.0
            self.rotation_y_keys[i] = 0.0
            self.rotation_z_keys[i] = 0.0
            self.strength_keys[i] = 0.45  # min(max(0.35 + np.sin(i / self.fps * 7) * 0.2, 0.05), 1)
            self.noise_keys[i] = 0.01
            self.prompt_keys[i] = (
                "a beautiful portrait of a diamond goddess with glittering skin and closed eyes, a detailed painting by greg rutkowski and raymond swanland, featured on cgsociety, fantasy art, detailed painting, artstation hd, photorealistic "
                if i < 15 * self.fps
                else "hyperrealistic dslr film still of jeff goldblum disguised legumes, beans, stunning 8 k octane comprehensive 3 d render, inspired by istvan sandorfi & greg rutkowski & unreal engine, perfect symmetry, dim volumetric cinematic lighting, extremely hyper - detailed, incredibly real lifelike attributes & flesh texture, intricate, masterpiece, artstation, stunning "
            )
            self.seed_keys[i] = i

        logger.info("Initializing model...")
        controlnet = CustomUNet2DConditionModel.from_pretrained(
            "takuma104/control_sd15_canny", subfolder="controlnet", torch_dtype=dtype
        )
        unet = CustomUNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet", torch_dtype=dtype)
        self.model = BlenderPipeline.from_pretrained(
            checkpoint,
            revision=revision,
            torch_dtype=dtype,
            controlnet=controlnet,
            unet=unet,
        )
        self.model = self.model.to(device)
        self.model.enable_xformers_memory_efficient_attention()
        self.blending = LatentBlending(self.model)

        self.wrap = FrameWrapper(
            self.translation_x_keys,
            self.translation_y_keys,
            self.translation_z_keys,
            self.rotation_x_keys,
            self.rotation_y_keys,
            self.rotation_z_keys,
        )

    def __prepare_target_latents(self):
        logger.info("Preparing target latents...")
        for frame in range(1, self.total_frames):
            if frame % self.target_latents_frames != 0:
                continue

            prompt = self.prompt_keys[frame]
            num_inference_steps = self.num_inference_steps_keys[frame]
            seed = self.seed_keys[frame]

            logger.info(f"Generating target latents for frame {frame}...")
            sample, images = self.model.g(
                prompt,
            )

            images[0].save(f"out/target_latents_{frame}.png")

            self.target_latents_keys[frame] = sample.detach().cpu().numpy()

    def test(self):
        # from vibrant_vision.sd.models.controlnet.model_loader import convert_controlnet_model

        # cn_path = "./models/diff_control_sd15_canny_fp16.safetensors"
        # config_path = "./models/cldm_v15.yaml"
        # convert_controlnet_model(config_path, cn_path, 512, False, controlnet_path)
        # return

        # img1 = np.array(Image.open("imgs/a.png"))
        # img2 = np.array(Image.open("imgs/b.png"))

        # for i, t in enumerate(rife_interpolate(img1, img2, 30)):
        #     Image.fromarray(t).save(f"imgs/{i}.png")

        # return
        pp2 = "1girl, (((skindentation))), cleavage, breasts, thigh strap, large breasts, breast press, earrings, no bra, very long hair, white hair, mini skirt, [fishnet thighhighs], thighs, thick thighs,  garter straps"
        pp = "nsfw,1 girl, (realistic:1.5),photorealistic, octane render,(hyperrealistic:1.2),(photorealistic face:1.2),sharp focus,soft lighting,(specular lighting:1.4), (Masterpiece), (Best Quality), fantasy, extremely detailed, intricate, hyper detailed,best quality,(8k), (4k),illustration, (perfect face),blue eyes,cute face,(aegyo sal:1),(((puffy eyes))), thick thighs,wide hips,huge thighs, standing,bridge,outdoor, scarf,white Jacket,white trousers, thick eyelashes,long eyelashes, brown hair,wavy hair,Coiled hair, dynamic pose,smiling,looking at viewer, (Kpop idol)"
        pp3 = "masterpiece, best quality, flat color, limited palette, low contrast, 1girl, serafuku, long straight black hair, lycoris flower, goat skull, (red, black)"
        np = "(worst quality, low quality:1.4), (realistic, lip, nose, tooth, rouge, lipstick, eyeshadow:1.0), (dusty sunbeams:1.0),, (abs, muscular, rib:1.0), (depth of field, bokeh, blurry:1.4), (greyscale, monochrome:1.0), text, title, logo, signature"
        list_prompts = []
        list_prompts.append(pp)
        list_prompts.append(pp)
        list_prompts.append(pp)
        list_prompts.append(pp2)
        list_prompts.append(pp2)
        list_prompts.append(pp3)

        fixed_seed = 1791072463
        variance_threshold = 0.65

        fp_movie = "./out/movie_example2.mp4"
        num_inference_steps = 20
        depth_strength = 0.65  # Specifies how deep (in terms of diffusion iterations the first branching happens)
        duration_single_trans = 2
        max_frames = 10

        self.blending.set_negative_prompt(np)
        self.blending.set_height(768)

        list_movie_parts = []
        for i in range(len(list_prompts) - 1):
            # For a multi transition we can save some computation time and recycle the latents
            if i == 0:
                self.blending.set_prompt1(list_prompts[i])
                self.blending.set_prompt2(list_prompts[i + 1])
                recycle_img1 = False
            else:
                self.blending.swap_forward()
                self.blending.set_prompt2(list_prompts[i + 1])
                recycle_img1 = True

            fp_movie_part = f"./out/tmp_part_{str(i).zfill(3)}.mp4"

            # Run latent blending
            frames = self.blending.run_transition(
                depth_strength=depth_strength,
                max_frames=max_frames,
                fixed_seed=fixed_seed,
                variance_threshold=variance_threshold,
                recycle_img1=recycle_img1,
                num_inference_steps=num_inference_steps,
            )

            logger.info(f"Created {len(frames)} frames for transition {i}")

            # Save movie
            self.blending.write_movie_transition(fp_movie_part, duration_single_trans, self.fps)
            list_movie_parts.append(fp_movie_part)

        movie.concatenate_movies(fp_movie, list_movie_parts)

    def perform(self):
        if self.with_target_latents:
            self.__prepare_target_latents()

        start_frame = 0
        end_frame = self.total_frames
        current_frame = start_frame
        cadence = 1

        prev_sample = None

        while current_frame < end_frame:
            prompt = self.prompt_keys[current_frame]
            num_inference_steps = self.num_inference_steps_keys[current_frame]
            seed = self.seed_keys[current_frame]
            noise = self.noise_keys[current_frame]
            strength = self.strength_keys[current_frame]

            target_latent_index = self.target_latents_keys[current_frame:].first_valid_index()
            last_target_latent_index = self.target_latents_keys[:current_frame].last_valid_index()
            if target_latent_index:
                target_latent = self.target_latents_keys[target_latent_index]
                target_latent = torch.from_numpy(target_latent).to(device)

            if prev_sample is not None:
                prev_sample = self.wrap.wrap_frame(
                    prev_sample,
                    current_frame,
                )

                # prev_sample = self.model.merge_noise(prev_sample, strength=noise)

            if prev_sample is None:
                prev_sample = self.blending.get_noise(seed)

            logger.info(
                f"Generating frame {current_frame}, strength={strength}, Tx={self.translation_x_keys[current_frame]}, Tz={self.translation_z_keys[current_frame]}..."
            )

            prompt_embeds = self.model.get_text_embeds(prompt)
            list_latents = self.model(prompt_embeds, latents=prev_sample)
            prev_sample = list_latents[-1]
            image = self.model.decode_latents(prev_sample)

            if target_latent_index is not None and last_target_latent_index is not None:
                t = (current_frame - last_target_latent_index) / (target_latent_index - last_target_latent_index)
                prev_sample = self.model.merge_latents(prev_sample, target_latent, t)

            Image.fromarray(image).save(f"out/image_{current_frame}.png")

            current_frame += cadence
