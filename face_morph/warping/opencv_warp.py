"""OpenCV-based optimized warping engine."""

import numpy as np
import cv2
from .base import WarpingEngine
from ..geometry.affine import compute_affine_transform


class OpenCVWarper(WarpingEngine):
    """Optimized warping using OpenCV's warpAffine.

    This is the recommended implementation for production use.
    It's ~10-50x faster than pure Python inverse mapping.
    """

    def warp(
        self,
        image: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
        triangles: list[tuple[int, int, int]],
        output_size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Warp image using OpenCV's optimized affine warping.

        Works with normalized coordinates where:
        - src_landmarks are in normalized space (height=1, centered)
        - dst_landmarks are in normalized space
        - Actual pixel sampling uses image dimensions
        """
        src_h, src_w = image.shape[:2]

        # Determine output size
        if output_size is None:
            out_w, out_h = src_w, src_h
        else:
            out_w, out_h = output_size

        # Convert normalized landmarks to pixel coordinates
        # src: normalized (height=1) -> src pixel coords
        src_pixels = src_landmarks.copy()
        src_pixels[:, 0] = src_landmarks[:, 0] * src_h + src_w / 2.0
        src_pixels[:, 1] = src_landmarks[:, 1] * src_h + src_h / 2.0

        # dst: normalized (height=1) -> output pixel coords
        dst_pixels = dst_landmarks.copy()
        dst_pixels[:, 0] = dst_landmarks[:, 0] * out_h + out_w / 2.0
        dst_pixels[:, 1] = dst_landmarks[:, 1] * out_h + out_h / 2.0

        output = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)

        for tri_indices in triangles:
            # Get triangle vertices in pixel coordinates
            src_tri = src_pixels[list(tri_indices)].astype(np.float32)
            dst_tri = dst_pixels[list(tri_indices)].astype(np.float32)

            # Compute affine transform: src -> dst
            warp_mat = compute_affine_transform(src_tri, dst_tri)

            # Create mask for destination triangle
            mask = np.zeros((out_h, out_w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_tri.astype(np.int32), 255)

            # Warp entire image
            warped = cv2.warpAffine(
                image,
                warp_mat,
                (out_w, out_h),
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Blend into output using mask
            for c in range(image.shape[2]):
                output[:, :, c] = np.where(
                    mask > 0,
                    warped[:, :, c],
                    output[:, :, c]
                )

        return output
