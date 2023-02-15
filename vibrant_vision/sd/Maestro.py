from pathlib import Path

import numpy as np
import torch

from vibrant_vision.general.logging import get_logger
from vibrant_vision.sd import constants
from vibrant_vision.sd.animation.keyframes import KeyFrames
from vibrant_vision.sd.animation.wrap import FrameWrapper
from vibrant_vision.sd.models.ImageModel import ImageModel

logger = get_logger(__name__)
device = constants.device


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
            self.translation_x_keys[i] = 0.0  # 10.0 * np.sin(2 * np.pi * i / self.fps)
            self.translation_y_keys[i] = 0.0
            self.translation_z_keys[i] = 1.01  # np.cos(i / self.fps * (2 / 3)) + i / self.fps * (1 / 5)
            self.rotation_x_keys[i] = 0.0
            self.rotation_y_keys[i] = 0.0
            self.rotation_z_keys[i] = 0.0
            self.strength_keys[i] = 0.45  # min(max(0.35 + np.sin(i / self.fps * 7) * 0.2, 0.05), 1)
            self.noise_keys[i] = 0.01
            self.prompt_keys[
                i
            ] = "a beautiful portrait of a diamond goddess with glittering skin and closed eyes, a detailed painting by greg rutkowski and raymond swanland, featured on cgsociety, fantasy art, detailed painting, artstation hd, photorealistic "

        logger.info("Initializing model...")
        self.model = ImageModel()
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

            logger.info(
                f"Generating frame {current_frame}, strength={strength}, Tx={self.translation_x_keys[current_frame]}, Tz={self.translation_z_keys[current_frame]}..."
            )
            prev_sample, images = self.model.g(
                prompt,
                sample=prev_sample,
                strength=strength,
            )

            if target_latent_index is not None and last_target_latent_index is not None:
                t = (current_frame - last_target_latent_index) / (target_latent_index - last_target_latent_index)
                prev_sample = self.model.merge_latents(prev_sample, target_latent, t)

            images[0].save(f"out/image_{current_frame}.png")

            current_frame += cadence
