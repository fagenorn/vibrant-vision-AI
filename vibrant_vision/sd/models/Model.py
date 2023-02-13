import torch
from diffusers import StableDiffusionPipeline
from vibrant_vision.sd import constants
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler                                                                                                                                                         

checkpoint = "runwayml/stable-diffusion-v1-5"
revision = "fp16"
dtype = torch.float16
device = constants.device

class Model:
    def __init__(self) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(checkpoint, torch_dtype=dtype, revision=revision)
        self.pipe = self.pipe.to(device)

        self.pipe.enable_xformers_memory_efficient_attention()     

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)     

    def get_inputs(self, batch_size=1, prompt=None):                                                                                                                                                                                                                 
        generator = [torch.Generator(device).manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
        prompts = batch_size * [prompt]                                                                                                                                                                                                             
        num_inference_steps = 20                                                                                                                                                                                                                    

        return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}  

    def predict(self, input):
        inputs = self.get_inputs(prompt=input, batch_size=4)
        return self.pipe(**inputs)
