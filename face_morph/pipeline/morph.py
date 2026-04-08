"""Face morphing pipeline with normalized coordinate system."""

import numpy as np
from ..geometry.delaunay import compute_delaunay_triangles
from ..warping.inverse_mapping import create_warper
from ..blending.alpha_blend import alpha_blend


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


def denormalize_landmarks(
    landmarks: np.ndarray,
    output_width: int,
    output_height: int,
    face_scale: float
) -> np.ndarray:
    """Convert normalized landmarks back to pixel coordinates.

    Args:
        landmarks: (N, 2) normalized coordinates
        output_width: target image width
        output_height: target image height
        face_scale: scale factor

    Returns:
        (N, 2) pixel coordinates centered in output image
    """
    pixel_coords = landmarks * face_scale
    pixel_coords[:, 0] += output_width / 2.0
    pixel_coords[:, 1] += output_height / 2.0
    return pixel_coords


def add_frame_points_normalized() -> np.ndarray:
    """Add frame anchor points in face-centered normalized coordinate system.

    Creates points that form a bounding rectangle around the face.

    Returns:
        (8, 2) array of normalized frame points
    """
    dist = 2.0  # Distance from face center in normalized units
    return np.array([
        [-dist, -dist],   # top-left
        [0, -dist],       # top-mid
        [dist, -dist],    # top-right
        [-dist, 0],       # left-mid
        [dist, 0],        # right-mid
        [-dist, dist],    # bottom-left
        [0, dist],        # bottom-mid
        [dist, dist],     # bottom-right
    ], dtype=np.float64)


def morph_faces(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: np.ndarray,
    landmarks2: np.ndarray,
    alpha: float = 0.5,
    warper: str = 'opencv',
    output_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Morph two faces at given alpha.

    Uses face-centered coordinate system to keep faces centered in output.

    Args:
        image1: First face image (H1, W1, 3)
        image2: Second face image (H2, W2, 3)
        landmarks1: (N, 2) landmarks for image1 in pixel coordinates
        landmarks2: (N, 2) landmarks for image2 in pixel coordinates
        alpha: Blending weight [0, 1]
        warper: 'opencv' or 'inverse'
        output_size: (width, height) for output. If None, uses image1's size.

    Returns:
        Morphed image with faces centered
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Determine output size
    if output_size is None:
        output_w, output_h = w1, h1
    else:
        output_w, output_h = output_size

    # Calculate face centers and scales
    center1, scale1 = calculate_face_center_and_scale(landmarks1)
    center2, scale2 = calculate_face_center_and_scale(landmarks2)

    # Compute mean scale for output
    scale_mean = (1 - alpha) * scale1 + alpha * scale2

    # Normalize landmarks to face-centered coordinate system
    lm1_norm = normalize_landmarks(landmarks1, center1, scale1)
    lm2_norm = normalize_landmarks(landmarks2, center2, scale2)

    # Compute mean shape in normalized space
    lm_mean_norm = (1 - alpha) * lm1_norm + alpha * lm2_norm

    # Add frame points for boundary coverage
    frame = add_frame_points_normalized()

    lm1_ext = np.vstack([lm1_norm, frame])
    lm2_ext = np.vstack([lm2_norm, frame])
    lm_mean_ext = np.vstack([lm_mean_norm, frame])

    # Compute Delaunay triangulation on mean shape
    triangles = compute_delaunay_triangles(lm_mean_ext)

    # Create warping engine
    warping_engine = create_warper(warper)

    # Destination face center is always the center of the output canvas
    # This ensures the face is centered in the output
    dst_face_center = np.array([output_w / 2.0, output_h / 2.0])

    # Warp both faces to mean shape
    warped1 = warping_engine.warp(
        image1, lm1_ext, lm_mean_ext, triangles,
        output_size=(output_w, output_h),
        src_face_center=center1,
        dst_face_center=dst_face_center,
        src_face_scale=scale1,
        dst_face_scale=scale_mean
    )
    warped2 = warping_engine.warp(
        image2, lm2_ext, lm_mean_ext, triangles,
        output_size=(output_w, output_h),
        src_face_center=center2,
        dst_face_center=dst_face_center,
        src_face_scale=scale2,
        dst_face_scale=scale_mean
    )

    # Blend
    result = alpha_blend(warped1, warped2, alpha)

    return result
