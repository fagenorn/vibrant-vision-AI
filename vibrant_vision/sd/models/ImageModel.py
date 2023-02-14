import numpy as np
import torch
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from vibrant_vision.sd import constants
from vibrant_vision.sd.models.pipeline import VVStableDiffusionPipeline

checkpoint = "runwayml/stable-diffusion-v1-5"
revision = "fp16"
dtype = torch.float16
device = constants.device


class ImageModel:
    def __init__(self) -> None:
        self.pipe = VVStableDiffusionPipeline.from_pretrained(checkpoint, torch_dtype=dtype, revision=revision)

        self.pipe = self.pipe.to(device)

        self.pipe.enable_xformers_memory_efficient_attention()

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.safety_checker = lambda images, **kwargs: (images, False)

    def g(self, prompt, latents=None, sample=None, strength=0.1):
        inputs = {
            "prompt": prompt,
            "num_inference_steps": 20,
            "return_sample": True,
            "generator": [torch.Generator(device).manual_seed(np.random.randint(0, 2**31 - 1))],
            "timesteps": None,
            "latents": None,
        }

        if sample is not None:
            self.pipe.scheduler.set_timesteps(inputs["num_inference_steps"], device=device)
            timesteps, inputs["num_inference_steps"] = self.__get_timesteps(inputs["num_inference_steps"], strength)
            latent_timestep = timesteps[:1].repeat(1)
            latents = self.__get_init_latents(
                sample,
                inputs["generator"],
            )
            latents = self.__add_noise(latents, strength, latent_timestep)

            inputs["timesteps"] = timesteps
            inputs["latents"] = latents

        return self.pipe(**inputs)

    def __add_noise(self, latents, strength, timestep):
        noise = torch.randn(latents.shape, device=device, dtype=dtype)
        latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

        return latents

    def __get_init_latents(self, sample, generator):
        vae = self.pipe.vae

        init_latents = vae.encode(sample).latent_dist.sample(generator)
        init_latents = 0.18215 * init_latents

        return torch.cat([init_latents], dim=0)

    def __get_timesteps(self, num_inference_steps, strength):
        timesteps = self.pipe.scheduler.timesteps
        init_timestep = min(int(strength * num_inference_steps), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
