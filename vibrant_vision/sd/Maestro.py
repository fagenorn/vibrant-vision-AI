from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, StableDiffusionControlNetPipeline, UNet2DConditionModel
from PIL import Image

import vibrant_vision.sd.animation.movie as movie
from vibrant_vision.general.logging import get_logger
from vibrant_vision.sd import constants
from vibrant_vision.sd.animation.blending import LatentBlending, rife_interpolate
from vibrant_vision.sd.animation.keyframes import KeyFrames
from vibrant_vision.sd.animation.wrap import FrameWrapper
from vibrant_vision.sd.models.controlnet.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from vibrant_vision.sd.models.ImageModel import BlenderPipeline
from vibrant_vision.sd.models.UpscalerModel import BlenderLatentUpscalePipeline
from safetensors import safe_open
from vibrant_vision.sd.models.deep_danbooru_model import DanbooruInterogator

logger = get_logger(__name__)
device = constants.device
checkpoint = "./models/aom3/"
revision = "main"
dtype = torch.float16
controlnet_checkpoint = "takuma104/control_sd15_canny"
upscaler_checkpoint = "stabilityai/sd-x2-latent-upscaler"


def load_learned_embed_in_clip_pt(learned_embeds_path, text_encoder, tokenizer, token):
    loaded_learned_embeds = {}
    with safe_open(learned_embeds_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            loaded_learned_embeds[key] = f.get_tensor(key)

    # print(loaded_learned_embeds.keys()  )
    # separate token and the embeds
    # string_to_param = loaded_learned_embeds["string_to_param"]
    # embeds = next(iter(string_to_param.items()))[1]
    embeds = loaded_learned_embeds["emb_params"]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    tokens = [token] * embeds.shape[0]

    # add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(tokens)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    for token_id, embed in zip(token_ids, embeds):
        text_encoder.get_input_embeddings().weight.data[token_id] = embed


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
        # controlnet = CustomUNet2DConditionModel.from_pretrained(
        #     controlnet_checkpoint, subfolder="controlnet", torch_dtype=dtype
        # )
        # unet = CustomUNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(
            checkpoint,
            subfolder="vae",
            torch_dtype=dtype,
        )
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint,
            subfolder="unet",
            torch_dtype=dtype,
        )
        self.model = BlenderPipeline.from_pretrained(
            controlnet_checkpoint,
            # revision=revision,
            torch_dtype=dtype,
            vae=vae,
            unet=unet,
        )

        load_learned_embed_in_clip_pt(
            f"embeds/easynegative.safetensors", self.model.text_encoder, self.model.tokenizer, f"<easynegative>"
        )

        # self.model.load_lora_checkpoint("./lora/openjourneyLora.safetensors", alpha=1.0)
        # self.model.load_lora_checkpoint("./lora/2bNierAutomataLora_v2b.safetensors", alpha=1.0)
        self.upscaler = BlenderLatentUpscalePipeline.from_pretrained(
            upscaler_checkpoint,
            torch_dtype=dtype,
        )

        self.blending = LatentBlending(self.model, self.upscaler)

        self.danbooru_interrogator = DanbooruInterogator("models/model-resnet_custom_v3.pt")

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

    def danbooru_blip(self, image: np.ndarray):
        image = Image.fromarray(image).convert("RGB")
        image = image.resize((512, 512))
        return self.danbooru_interrogator.predict(image)

    def bad_apple(self):
        import cv2

        logger.info("Starting bad apple...")

        # open video
        video = cv2.VideoCapture("bad_apple.mp4")
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video fps: {fps}")
        logger.info(f"Video total frames: {total_frames}")

        # Read frames in 1 second intervals
        ret = True

        while ret:
            frames = []
            for i in range(int(fps)):
                ret, frame = video.read()

                if not ret:
                    break

                frames.append(frame)

            if len(frames) < 3:
                break

            first_frame = frames[0]
            last_frame = frames[-1]

            first_prompt = self.danbooru_blip(first_frame)
            last_prompt = self.danbooru_blip(last_frame)

            logger.info(f"First prompt: {first_prompt}")
            logger.info(f"Last prompt: {last_prompt}")

            cv2.imshow("first", first_frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(10000) & 0xFF == ord("q"):
                break

    def test(self):
        # from vibrant_vision.sd.models.controlnet.model_loader import convert_controlnet_model

        # cn_path = "./models/diff_control_sd15_canny_fp16.safetensors"
        # config_path = "./models/cldm_v15.yaml"
        # convert_controlnet_model(config_path, cn_path, 512, False, controlnet_path)
        # return

        # import numpy as np
        # from vibrant_vision.sd.animation.rife.RIFE import RIFE

        # img1 = np.array(Image.open("imgs/a.png"))
        # img2 = np.array(Image.open("imgs/b.png"))

        # for i, t in enumerate(RIFE(img1, img2, 30)):
        #     Image.fromarray(t).save(f"imgs/{i}.png")

        # return
        pp = "masterpiece, best quality, 1girl, solo, long_hair, breasted, highres, blush, smile, looking_at_viewer, hair_ornament, bow, twintails, brown_eyes, cleavage, white_background, school_uniform, sitting, animal_ears, green_eyes, holding, shoes, heart, striped, elbow_gloves"
        pp1 = "masterpiece, best quality, 1girl, solo, long_hair, breasted, highres, blush, smile, looking_at_viewer, hair_ornament, bow, twintails, brown_eyes, cleavage, white_background, school_uniform, sitting, animal_ears, green_eyes, holding, shoes, heart, striped, elbow_gloves, choker, pink_eyes, covered_nipples, arms_up, collar, dutch_angle, book, vest, kneehighs, one_eye_closed, looking_back"
        pp2 = "masterpiece, best quality, 1girl, solo, long_hair, breasted, highres, blush, smile, looking_at_viewer, hair_ornament, bow, twintails, brown_eyes, cleavage, white_background, school_uniform, sitting, animal_ears, green_eyes, holding, shoes, heart, striped, elbow_gloves, choker, pink_eyes, covered_nipples, arms_up, collar, dutch_angle, book, vest, kneehighs, one_eye_closed, looking_back, food, small_breasts, white_legwear, pleated_skirt, frills, flower, yellow_eyes, pink_hair, wings, hair_bow"
        np = "easynegative, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, day, daylight, blue sky, white background"

        list_prompts = []
        list_prompts.append(pp)
        list_prompts.append(pp)
        list_prompts.append(pp1)
        list_prompts.append(pp1)
        list_prompts.append(pp2)
        list_prompts.append(pp2)

        fixed_seed = 1791072463
        variance_threshold = 0.45

        fp_movie = "./out/movie_example2.mp4"
        num_inference_steps = 10
        depth_strength = 0.65  # Specifies how deep (in terms of diffusion iterations the first branching happens)
        duration_single_trans = 1
        max_frames = 15

        self.blending.set_negative_prompt(np)
        self.blending.set_controlnet_guidance_percent(0.8)
        self.blending.set_guidance_scale(7.0)

        # self.blending.set_height(768)

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
