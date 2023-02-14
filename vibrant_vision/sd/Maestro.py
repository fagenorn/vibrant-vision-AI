from pathlib import Path

import numpy as np
import torch

from vibrant_vision.general.logging import get_logger
from vibrant_vision.sd.animation.keyframes import KeyFrames
from vibrant_vision.sd.animation.wrap import FrameWrapper
from vibrant_vision.sd.models.ImageModel import ImageModel

logger = get_logger(__name__)


class Maestro:
    models_path = Path(__file__).parent / "models"
    configs_path = Path(__file__).parent / "configs"
    output_path = Path(__file__).parent / "output"

    def __init__(self, fps) -> None:
        self.fps = fps
        self.total_frames = 15 * self.fps  # Hardcoded duration for now: 15 seconds at 30 fps = 450 frames

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

        for i in range(self.total_frames):
            self.translation_x_keys[i] = 10.0
            self.translation_y_keys[i] = 0.0
            self.translation_z_keys[i] = 10.0
            self.rotation_x_keys[i] = 0.0
            self.rotation_y_keys[i] = 0.0
            self.rotation_z_keys[i] = 0.0
            self.strength_keys[i] = 0.5

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

    def perform(self):
        start_frame = 0
        end_frame = 30
        current_frame = start_frame
        cadence = 1

        prev_sample = None

        while current_frame < end_frame:
            translation_x = self.translation_x_keys[current_frame]
            translation_y = self.translation_y_keys[current_frame]
            translation_z = self.translation_z_keys[current_frame]
            rotation_x = self.rotation_x_keys[current_frame]
            rotation_y = self.rotation_y_keys[current_frame]
            rotation_z = self.rotation_z_keys[current_frame]

            if prev_sample is not None:
                prev_sample, depth = self.wrap.wrap_frame(
                    prev_sample,
                    current_frame,
                )

            prompt = self.prompt_keys[current_frame]
            num_inference_steps = self.num_inference_steps_keys[current_frame]
            seed = self.seed_keys[current_frame]

            logger.info(f"Generating frame {current_frame}...")
            prev_sample, images = self.model.g("a photo of a cute bear", sample=prev_sample)

            images[0].save(f"out/image_{current_frame}.png")

            current_frame += cadence
