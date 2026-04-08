"""Face morphing pipeline with normalized coordinate system."""

import numpy as np
from ..geometry.delaunay import compute_delaunay_triangles
from ..warping.inverse_mapping import create_warper
from ..blending.alpha_blend import alpha_blend


def normalize_landmarks(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize landmarks to centered coordinate system.

    Maps pixel coordinates to normalized space where:
    - Origin (0, 0) is at image center
    - Y ranges from -0.5 to 0.5 (height = 1)
    - X ranges based on aspect ratio (width = aspect_ratio)

    Args:
        landmarks: (N, 2) pixel coordinates
        width: image width in pixels
        height: image height in pixels

    Returns:
        (N, 2) normalized coordinates
    """
    # Convert to centered coordinates (origin at center)
    centered = landmarks.copy()
    centered[:, 0] = landmarks[:, 0] - width / 2.0   # x - center_x
    centered[:, 1] = landmarks[:, 1] - height / 2.0  # y - center_y

    # Normalize so height = 1 (y in [-0.5, 0.5])
    normalized = centered / height

    return normalized


def denormalize_landmarks(
    landmarks: np.ndarray,
    output_width: int,
    output_height: int
) -> np.ndarray:
    """Convert normalized landmarks back to pixel coordinates.

    Args:
        landmarks: (N, 2) normalized coordinates
        output_width: target image width
        output_height: target image height

    Returns:
        (N, 2) pixel coordinates
    """
    # Scale by output height
    pixel_coords = landmarks * output_height

    # Shift to center
    pixel_coords[:, 0] += output_width / 2.0
    pixel_coords[:, 1] += output_height / 2.0

    return pixel_coords


def add_frame_points_normalized(aspect_ratio: float) -> np.ndarray:
    """Add frame anchor points in normalized coordinate system.

    Creates points that form a bounding rectangle around the normalized space.

    Args:
        aspect_ratio: width / height ratio

    Returns:
        (8, 2) array of normalized frame points
    """
    # Half dimensions
    half_w = aspect_ratio / 2.0  # x ranges from -half_w to +half_w
    half_h = 0.5                 # y ranges from -half_h to +half_h

    # 8 points: corners + edge midpoints
    frame_points = np.array([
        [-half_w, -half_h],  # top-left
        [0, -half_h],        # top-mid
        [half_w, -half_h],   # top-right
        [-half_w, 0],        # left-mid
        [half_w, 0],         # right-mid
        [-half_w, half_h],   # bottom-left
        [0, half_h],         # bottom-mid
        [half_w, half_h],    # bottom-right
    ], dtype=np.float64)

    return frame_points


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

    Uses normalized coordinate system to handle images of different sizes
    without distortion.

    Args:
        image1: First face image (H1, W1, 3)
        image2: Second face image (H2, W2, 3)
        landmarks1: (N, 2) landmarks for image1 in pixel coordinates
        landmarks2: (N, 2) landmarks for image2 in pixel coordinates
        alpha: Blending weight [0, 1]
        warper: 'opencv' or 'inverse'
        output_size: (width, height) for output. If None, uses image1's size.

    Returns:
        Morphed image
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Determine output size
    if output_size is None:
        output_w, output_h = w1, h1
    else:
        output_w, output_h = output_size

    # Calculate aspect ratios
    aspect1 = w1 / h1
    aspect2 = w2 / h2
    aspect_mean = (1 - alpha) * aspect1 + alpha * aspect2

    # Normalize landmarks to centered coordinate system
    lm1_norm = normalize_landmarks(landmarks1, w1, h1)
    lm2_norm = normalize_landmarks(landmarks2, w2, h2)

    # Compute mean shape in normalized space
    lm_mean_norm = (1 - alpha) * lm1_norm + alpha * lm2_norm

    # Add frame points for boundary coverage (in normalized space)
    frame1 = add_frame_points_normalized(aspect1)
    frame2 = add_frame_points_normalized(aspect2)
    frame_mean = add_frame_points_normalized(aspect_mean)

    lm1_ext = np.vstack([lm1_norm, frame1])
    lm2_ext = np.vstack([lm2_norm, frame2])
    lm_mean_ext = np.vstack([lm_mean_norm, frame_mean])

    # Compute Delaunay triangulation on mean shape (normalized)
    triangles = compute_delaunay_triangles(lm_mean_ext)

    # Create warping engine
    warping_engine = create_warper(warper)

    # Warp both faces to mean shape
    # Pass output size explicitly so both warpers use the same canvas
    warped1 = warping_engine.warp(
        image1, lm1_ext, lm_mean_ext, triangles,
        output_size=(output_w, output_h)
    )
    warped2 = warping_engine.warp(
        image2, lm2_ext, lm_mean_ext, triangles,
        output_size=(output_w, output_h)
    )

    # Blend
    result = alpha_blend(warped1, warped2, alpha)

    return result
