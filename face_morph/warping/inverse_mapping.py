"""Pure Python inverse mapping warping engine."""

import numpy as np
import cv2
from .base import WarpingEngine
from ..geometry.affine import compute_inverse_transform, invert_affine_matrix
from ..geometry.barycentric import point_in_triangle, compute_barycentric_coords
from ..geometry.delaunay import get_triangle_bounding_box


class InverseMappingWarper(WarpingEngine):
    """Pure Python implementation of inverse warping.

    This implementation follows the tutorial exactly:
    1. For each triangle, compute inverse affine transform
    2. Iterate over bounding box pixels
    3. Test point-in-triangle using barycentric coordinates
    4. Back-project to source and sample with bilinear interpolation

    This is slower than OpenCVWarper but demonstrates the algorithm clearly.
    """

    def warp(
        self,
        image: np.ndarray,
        src_landmarks: np.ndarray,
        dst_landmarks: np.ndarray,
        triangles: list[tuple[int, int, int]],
        output_size: tuple[int, int] | None = None,
        src_face_center: np.ndarray = None,
        dst_face_center: np.ndarray = None,
        src_face_scale: float = 1.0,
        dst_face_scale: float = 1.0
    ) -> np.ndarray:
        """Warp image using pure Python inverse mapping.

        Works with face-centered normalized coordinates.
        """
        src_h, src_w = image.shape[:2]

        # Determine output size
        if output_size is None:
            out_w, out_h = src_w, src_h
        else:
            out_w, out_h = output_size

        # Default face centers if not provided
        if src_face_center is None:
            src_face_center = np.array([src_w / 2.0, src_h / 2.0])
        if dst_face_center is None:
            dst_face_center = np.array([out_w / 2.0, out_h / 2.0])

        # Convert normalized landmarks to pixel coordinates
        # src: normalized (face-centered) -> src pixel coords
        src_pixels = src_landmarks.copy() * src_face_scale
        src_pixels[:, 0] += src_face_center[0]
        src_pixels[:, 1] += src_face_center[1]

        # dst: normalized (face-centered) -> output pixel coords
        dst_pixels = dst_landmarks.copy() * dst_face_scale
        dst_pixels[:, 0] += dst_face_center[0]
        dst_pixels[:, 1] += dst_face_center[1]

        output = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
        filled = np.zeros((out_h, out_w), dtype=bool)

        for tri_indices in triangles:
            # Get triangle vertices in pixel coordinates
            src_tri = src_pixels[list(tri_indices)].astype(np.float64)
            dst_tri = dst_pixels[list(tri_indices)].astype(np.float64)

            # Compute inverse affine transform: dst -> src
            inv_mat = compute_inverse_transform(src_tri, dst_tri)

            # Get bounding box of destination triangle
            x_min, x_max, y_min, y_max = get_triangle_bounding_box(dst_tri, out_w, out_h)

            # Iterate over pixels in bounding box
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if filled[y, x]:
                        continue

                    point = np.array([x, y], dtype=np.float64)

                    # Check if point is inside triangle
                    if not point_in_triangle(point, dst_tri):
                        continue

                    # Back-project to source coordinates
                    ones = np.array([1])
                    point_h = np.hstack([point, ones])
                    src_coord = inv_mat @ point_h
                    x_s, y_s = src_coord[0], src_coord[1]

                    # Sample source with bilinear interpolation
                    color = self._bilinear_sample(image, x_s, y_s)

                    # Write to output
                    output[y, x] = color
                    filled[y, x] = True

        return output

    def _bilinear_sample(self, image: np.ndarray, x: float, y: float) -> np.ndarray:
        """Sample image at (x, y) using bilinear interpolation."""
        h, w = image.shape[:2]

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        fx = x - x0
        fy = y - y0

        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))

        c00 = image[y0, x0]
        c10 = image[y0, x1]
        c01 = image[y1, x0]
        c11 = image[y1, x1]

        color = (
            (1 - fx) * (1 - fy) * c00 +
            fx * (1 - fy) * c10 +
            (1 - fx) * fy * c01 +
            fx * fy * c11
        )

        return color.astype(image.dtype)


def create_warper(name: str) -> WarpingEngine:
    """Factory function to create warping engine."""
    if name == 'opencv':
        from .opencv_warp import OpenCVWarper
        return OpenCVWarper()
    elif name == 'inverse':
        return InverseMappingWarper()
    else:
        raise ValueError(f"Unknown warper: {name}")
