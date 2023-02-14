import math

import numpy as np
import torch
from einops import rearrange

import vibrant_vision.sd.animation.py3d_tools as p3d
from vibrant_vision.sd import constants
from vibrant_vision.sd.models.DepthModel import DepthModel
import cv2

TRANSLATION_SCALE = 1.0 / 200.0
device = constants.device


class FrameWrapper:
    def __init__(self, trans_x_keys, trans_y_keys, trans_z_keys, rot_x_keys, rot_y_keys, rot_z_keys):
        self.trans_x_keys = trans_x_keys
        self.trans_y_keys = trans_y_keys
        self.trans_z_keys = trans_z_keys
        self.rot_x_keys = rot_x_keys
        self.rot_y_keys = rot_y_keys
        self.rot_z_keys = rot_z_keys

        self.depth_model = DepthModel()

    @staticmethod
    def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
        sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
        sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
        sample_int8 = sample_f32 * 255
        return sample_int8.astype(type)

    @staticmethod
    def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
        sample = ((sample.astype(float) / 255.0) * 2) - 1
        sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
        sample = torch.from_numpy(sample)

        return sample

    def wrap_frame(self, sample, frame):
        if not isinstance(sample, np.ndarray):
            sample = FrameWrapper.sample_to_cv2(sample)

        depth = self.__predict_depth(sample)

        cv2.imwrite("out/sample_1.png", sample)

        sample = self.__wrap_image(sample, depth, frame)

        cv2.imwrite("out/sample_2.png", sample)

        sample = FrameWrapper.sample_from_cv2(sample)
        sample = sample.half().to(device)

        return sample, depth

    def __unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def __predict_depth(self, image):
        depth = self.depth_model.predict(image)
        depth = depth.squeeze().cpu().numpy()

        # MiDaS makes the near values greater, and the far values lesser.
        depth = np.subtract(50.0, depth)
        depth = depth / 19.0

        depth = np.expand_dims(depth, axis=0)
        return torch.from_numpy(depth).squeeze().to(device)

    def __wrap_image(self, image, depth_tensor, frame):
        trans_xyz = [
            -self.trans_x_keys[frame] * TRANSLATION_SCALE,
            self.trans_y_keys[frame] * TRANSLATION_SCALE,
            -self.trans_z_keys[frame] * TRANSLATION_SCALE,
        ]

        rot_xyz = [
            math.radians(self.rot_x_keys[frame]),
            math.radians(self.rot_y_keys[frame]),
            math.radians(self.rot_z_keys[frame]),
        ]

        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rot_xyz, device=device), "XYZ").unsqueeze(0)

        w, h = image.shape[:2]

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

        z = depth_tensor

        xyz_world_old = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
        xyz_cam_xy_old = persp_cam_old.get_full_projection_transform().transform_points(xyz_world_old)[:, 0:2]
        xyz_cam_xy_new = persp_cam_new.get_full_projection_transform().transform_points(xyz_world_old)[:, 0:2]

        offset_xy = xyz_cam_xy_new - xyz_cam_xy_old
        identity_2d_batch = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0)
        coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
        offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

        image_tensor = rearrange(torch.from_numpy(image.astype(np.float32)), "h w c -> c h w").to(device)
        wrapped_image = torch.nn.functional.grid_sample(
            image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
            offset_coords_2d,
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        )

        return rearrange(wrapped_image.squeeze().clamp(0, 255), "c h w -> h w c").cpu().numpy().astype(image.dtype)
