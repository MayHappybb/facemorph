"""Face morphing pipeline with normalized coordinate system."""

import numpy as np
from ..geometry.delaunay import compute_delaunay_triangles
from ..warping.inverse_mapping import create_warper
from ..blending.alpha_blend import multi_blend


def calculate_face_center_and_scale(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate face centroid and scale from landmarks.

    Args:
        landmarks: (N, 2) pixel coordinates

    Returns:
        (face_center, face_scale) where face_center is (2,) and face_scale is float
    """
    face_center = np.mean(landmarks, axis=0)
    centered = landmarks - face_center
    face_scale = np.mean(np.abs(centered)) * 2.0
    if face_scale < 1e-6:
        face_scale = 100.0  # Fallback
    return face_center, face_scale


def normalize_landmarks(landmarks: np.ndarray, face_center: np.ndarray, face_scale: float) -> np.ndarray:
    """Normalize landmarks to face-centered coordinate system.

    Maps pixel coordinates to normalized space where:
    - Origin (0, 0) is at the face centroid
    - Coordinates are scaled by face_scale

    Args:
        landmarks: (N, 2) pixel coordinates
        face_center: (2,) face centroid in pixels
        face_scale: scale factor for normalization

    Returns:
        (N, 2) normalized coordinates
    """
    return (landmarks - face_center) / face_scale


def calculate_intersection_canvas(
    images: list[np.ndarray],
    landmarks: list[np.ndarray],
    scale_mean: float
) -> tuple[int, int]:
    """Calculate output canvas as intersection of all face-centered images.

    IMPORTANT: Uses normalized space (face-centered, face-scaled) so face
    sizes are normalized to comparable scale before intersection calculation.

    Args:
        images: List of input images
        landmarks: List of landmark arrays
        scale_mean: Mean face scale for pixel conversion

    Returns:
        (width, height) in pixels, using scale_mean for conversion

    Raises:
        ValueError: If images have no common intersection region
    """
    bounds_list = []

    for img, lm in zip(images, landmarks):
        h, w = img.shape[:2]
        center, scale = calculate_face_center_and_scale(lm)

        # Image corners in pixel coordinates
        corners_pixel = np.array([
            [0, 0],           # top-left
            [w, 0],           # top-right
            [w, h],           # bottom-right
            [0, h],           # bottom-left
        ])

        # Convert to normalized (face-centered, face-scaled) space
        corners_norm = (corners_pixel - center) / scale

        # Bounds in normalized space
        x_min = corners_norm[:, 0].min()  # left edge
        x_max = corners_norm[:, 0].max()  # right edge
        y_min = corners_norm[:, 1].min()  # top edge
        y_max = corners_norm[:, 1].max()  # bottom edge

        bounds_list.append((x_min, x_max, y_min, y_max))

    # Intersection in normalized space
    x_min = max(b[0] for b in bounds_list)
    x_max = min(b[1] for b in bounds_list)
    y_min = max(b[2] for b in bounds_list)
    y_max = min(b[3] for b in bounds_list)

    # Validate intersection exists
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"Images have no common intersection region. "
            f"Bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]"
        )

    # Convert intersection bounds to pixels using mean scale
    # This ensures all faces appear at similar size in output
    width = int((x_max - x_min) * scale_mean)
    height = int((y_max - y_min) * scale_mean)

    return width, height


def create_frame_points_from_intersection(
    width: int,
    height: int,
    scale_mean: float
) -> np.ndarray:
    """Create frame points at canvas corners in normalized coords.

    These points ensure triangulation covers the full output canvas.

    Args:
        width: Output canvas width in pixels
        height: Output canvas height in pixels
        scale_mean: Mean face scale for normalization

    Returns:
        (8, 2) array of frame points in normalized space
    """
    half_w = width / 2.0
    half_h = height / 2.0

    # 8 points: corners + edge midpoints, in normalized space
    frame = np.array([
        [-half_w / scale_mean, -half_h / scale_mean],   # top-left
        [0, -half_h / scale_mean],                       # top-mid
        [half_w / scale_mean, -half_h / scale_mean],    # top-right
        [-half_w / scale_mean, 0],                       # left-mid
        [half_w / scale_mean, 0],                        # right-mid
        [-half_w / scale_mean, half_h / scale_mean],    # bottom-left
        [0, half_h / scale_mean],                        # bottom-mid
        [half_w / scale_mean, half_h / scale_mean],     # bottom-right
    ], dtype=np.float64)

    return frame


def morph_faces(
    images: list[np.ndarray],
    landmarks: list[np.ndarray],
    weights: list[float],
    warper: str = 'opencv',
    output_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Morph N faces at given weights.

    Uses face-centered coordinate system to keep faces centered in output
    and calculates canvas size as intersection of all face-centered images.

    Args:
        images: List of N face images [(H1, W1, 3), ...]
        landmarks: List of N landmark arrays [(68, 2), ...] in pixel coordinates
        weights: List of N blending weights, should sum to 1.0
        warper: 'opencv' or 'inverse'
        output_size: (width, height) for output. If None, uses intersection.

    Returns:
        Morphed image with faces centered
    """
    n = len(images)
    assert n == len(landmarks), "Number of images must match number of landmarks"
    assert n == len(weights), "Number of images must match number of weights"
    assert n >= 2, "Need at least 2 images to morph"

    # Normalize weights to sum to 1 if needed
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        weights = [w / total for w in weights]

    # Calculate face centers and scales for each image
    centers_scales = [calculate_face_center_and_scale(lm) for lm in landmarks]
    centers = [cs[0] for cs in centers_scales]
    scales = [cs[1] for cs in centers_scales]

    # Calculate mean scale (weighted)
    scale_mean = sum(w * s for w, s in zip(weights, scales))

    # Normalize all landmarks to face-centered coordinate system
    lm_norm_list = [
        normalize_landmarks(lm, c, s)
        for lm, c, s in zip(landmarks, centers, scales)
    ]

    # Compute weighted mean shape in normalized space
    lm_mean_norm = np.zeros_like(lm_norm_list[0])
    for w, lm in zip(weights, lm_norm_list):
        lm_mean_norm += w * lm

    # Determine output size (intersection if not specified)
    if output_size is None:
        output_w, output_h = calculate_intersection_canvas(images, landmarks, scale_mean)
    else:
        output_w, output_h = output_size

    # Add frame points at intersection bounds
    frame = create_frame_points_from_intersection(output_w, output_h, scale_mean)
    lm_ext_list = [np.vstack([lm, frame]) for lm in lm_norm_list]
    lm_mean_ext = np.vstack([lm_mean_norm, frame])

    # Compute Delaunay triangulation on mean shape
    triangles = compute_delaunay_triangles(lm_mean_ext)

    # Create warping engine
    warping_engine = create_warper(warper)

    # Destination face center is always the center of the output canvas
    dst_face_center = np.array([output_w / 2.0, output_h / 2.0])

    # Warp all images to mean shape
    warped_images = []
    for img, lm_ext, center, scale in zip(images, lm_ext_list, centers, scales):
        warped = warping_engine.warp(
            img, lm_ext, lm_mean_ext, triangles,
            output_size=(output_w, output_h),
            src_face_center=center,
            dst_face_center=dst_face_center,
            src_face_scale=scale,
            dst_face_scale=scale_mean
        )
        warped_images.append(warped)

    # N-way blend
    result = multi_blend(warped_images, weights)

    return result