"""
@source: https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
"""
__version__ = "3.1.0"


import imp
import torch
from threading import Lock
from .IFNet_HDv3_v4_6 import IFNet
import os
from functorch.compile import memory_efficient_fusion
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm

package_dir = os.path.dirname(os.path.realpath(__file__))
model_name = "flownet_v4.6.pkl"


@torch.inference_mode()
def RIFE(img1, img2, total_frames, scale=1.0):
    num_streams = 3
    ensemble = False
    device = torch.device("cuda")

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    checkpoint = torch.load(os.path.join(package_dir, model_name), map_location="cpu")
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items() if "module." in k}

    flownet = IFNet(scale, ensemble).half()
    flownet.load_state_dict(checkpoint, strict=False)
    flownet.eval().to(device, memory_format=torch.channels_last)

    h, w, _ = img1.shape
    tmp = max(128, int(128 * scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    flownet = memory_efficient_fusion(flownet)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(img0, img1, factor):
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            I0 = img_to_tensor(img0, device)
            I1 = img_to_tensor(img1, device)
            I0 = F.pad(I0, padding)
            I1 = F.pad(I1, padding)

            timestep = torch.full((1, 1, I0.shape[2], I0.shape[3]), factor, device=device).half()
            timestep = timestep.to(memory_format=torch.channels_last)

            output = flownet(I0, I1, timestep)

            return tensor_to_img(output[:, :, :h, :w], h, w)

    frames = []
    for i in tqdm(range(total_frames), desc="RIFE"):
        factor = 0.5 if total_frames == 1 else i / (total_frames - 1)
        frames.append(inference(img1, img2, factor))

    return frames


def img_to_tensor(img, device: torch.device) -> torch.Tensor:
    img = np.transpose(img, (2, 0, 1))
    array = torch.from_numpy(img).unsqueeze(0).to(device, memory_format=torch.channels_last).half()
    array /= 255.0
    return array


def tensor_to_img(tensor: torch.Tensor, h, w):
    img = tensor.squeeze(0).cpu().numpy()
    img *= 255.0
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = img[:h, :w]

    return img
