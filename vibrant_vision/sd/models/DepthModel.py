import torch
from transformers import AutoImageProcessor, DPTForDepthEstimation

from vibrant_vision.sd import constants

checkpoint = "Intel/dpt-hybrid-midas"
revision = "main"
dtype = torch.float32
device = constants.device


class DepthModel:
    def __init__(self) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained(checkpoint, torch_dtype=dtype, revision=revision)
        self.model = DPTForDepthEstimation.from_pretrained(checkpoint, torch_dtype=dtype, revision=revision)

    def predict(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[-2::-1],
            mode="bicubic",
            align_corners=False,
        )

        return prediction
