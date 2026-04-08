"""Alpha blending utilities."""

import numpy as np


def alpha_blend(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Blend two images using alpha compositing.

    Args:
        image1: First image (H, W, C) or (H, W)
        image2: Second image (H, W, C) or (H, W)
        alpha: Blending weight [0, 1]
            0.0 = pure image1
            0.5 = average
            1.0 = pure image2

    Returns:
        Blended image
    """
    # Convert to float for blending
    img1_f = image1.astype(np.float64)
    img2_f = image2.astype(np.float64)

    # Linear interpolation
    blended = (1 - alpha) * img1_f + alpha * img2_f

    # Convert back to original dtype
    if image1.dtype == np.uint8:
        blended = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        blended = blended.astype(image1.dtype)

    return blended
