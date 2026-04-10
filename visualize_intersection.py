#!/usr/bin/env python3
"""Diagnostic script to visualize intersection of face-centered images."""

import numpy as np
import cv2
from face_morph.landmarks.base import create_detector
from face_morph.pipeline.morph import calculate_face_center_and_scale, calculate_intersection_canvas


def visualize_intersection(image_paths: list[str], output_path: str = "intersection_visual.png"):
    """Create visualization showing all face-centered images superimposed.

    Each image is re-centered so face is at canvas center and scaled to common size.
    This reveals the intersection region that all images can cover.
    """
    # Load images and detect landmarks
    detector = create_detector("mediapipe")

    images = []
    landmarks = []
    for path in image_paths:
        img = cv2.imread(path)
        images.append(img)
        lm = detector.detect(img)
        landmarks.append(lm)
        print(f"Loaded {path}: {img.shape[1]}x{img.shape[0]}")

    # Calculate face centers and scales
    centers_scales = [calculate_face_center_and_scale(lm) for lm in landmarks]
    centers = [cs[0] for cs in centers_scales]
    scales = [cs[1] for cs in centers_scales]

    # Calculate mean scale
    scale_mean = np.mean(scales)

    # Calculate intersection canvas
    canvas_w, canvas_h = calculate_intersection_canvas(images, landmarks, scale_mean)
    print(f"\nIntersection canvas: {canvas_w}x{canvas_h}")

    # Calculate intersection bounds in normalized space
    bounds_list = []
    for i, (img, lm) in enumerate(zip(images, landmarks)):
        h, w = img.shape[:2]
        center, scale = calculate_face_center_and_scale(lm)

        corners_pixel = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        corners_norm = (corners_pixel - center) / scale

        bounds_list.append({
            'x_min': corners_norm[:, 0].min(),
            'x_max': corners_norm[:, 0].max(),
            'y_min': corners_norm[:, 1].min(),
            'y_max': corners_norm[:, 1].max(),
            'name': image_paths[i]
        })

    # Print bounds for each image
    print("\nNormalized bounds (face-centered, face-scaled):")
    for i, b in enumerate(bounds_list):
        print(f"  [{i}] {b['name']}: x=[{b['x_min']:.2f}, {b['x_max']:.2f}], y=[{b['y_min']:.2f}, {b['y_max']:.2f}]")

    # Intersection
    x_min = max(b['x_min'] for b in bounds_list)
    x_max = min(b['x_max'] for b in bounds_list)
    y_min = max(b['y_min'] for b in bounds_list)
    y_max = min(b['y_max'] for b in bounds_list)
    print(f"\nIntersection: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

    # Find which image limits the intersection most
    print("\nIntersection limits:")
    print(f"  x_min limited by: {[b['name'] for b in bounds_list if b['x_min'] == x_min][0]}")
    print(f"  x_max limited by: {[b['name'] for b in bounds_list if b['x_max'] == x_max][0]}")
    print(f"  y_min limited by: {[b['name'] for b in bounds_list if b['y_min'] == y_min][0]}")
    print(f"  y_max limited by: {[b['name'] for b in bounds_list if b['y_max'] == y_max][0]}")

    # For each image, shift so face center is at canvas center
    # and scale so face scale matches mean scale
    canvas_center = np.array([canvas_w / 2.0, canvas_h / 2.0])

    shifted_images = []
    for img, center, scale in zip(images, centers, scales):
        # Create output canvas
        shifted = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Transformation: from source pixels to output pixels
        # We want: src_face_center -> canvas_center
        # And: src_scale -> scale_mean (scale factor = scale_mean / scale)

        scale_factor = scale_mean / scale

        # Build affine matrix
        # First scale, then translate
        # In output coords: (pixel - canvas_center) / scale_mean = (src_pixel - center) / scale
        # So: src_pixel = (pixel - canvas_center) * scale / scale_mean + center

        # For warpAffine (forward mapping), we need: src -> dst
        # But we want inverse mapping: dst -> src
        # So we build the inverse transform matrix

        # Transformation:
        # x_src = (x_dst - canvas_center_x) * scale / scale_mean + center_x
        # y_src = (y_dst - canvas_center_y) * scale / scale_mean + center_y
        #
        # In affine matrix form (2x3):
        # [scale/scale_mean, 0, center_x - canvas_center_x * scale/scale_mean]
        # [0, scale/scale_mean, center_y - canvas_center_y * scale/scale_mean]

        # Actually, for cv2.warpAffine we need forward mapping: src -> dst
        # dst = T @ src
        # x_dst = (x_src - center_x) * scale_mean / scale + canvas_center_x
        # y_dst = (y_src - center_y) * scale_mean / scale + canvas_center_y

        s = scale_mean / scale
        M = np.array([
            [s, 0, canvas_center[0] - center[0] * s],
            [0, s, canvas_center[1] - center[1] * s]
        ], dtype=np.float32)

        # Warp with inverse mapping for proper sampling
        warped = cv2.warpAffine(
            img, M, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        shifted_images.append(warped)

    # Blend all shifted images with equal weight
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    for shifted in shifted_images:
        result += (1.0 / len(shifted_images)) * shifted.astype(np.float64)

    result = np.clip(result, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"\nSaved intersection visualization to {output_path}")

    # Also save individual shifted images for debugging
    for i, shifted in enumerate(shifted_images):
        cv2.imwrite(f"shifted_{i}.png", shifted)
        print(f"Saved shifted_{i}.png")

    return bounds_list, (x_min, x_max, y_min, y_max)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] if len(sys.argv) > 1 else [
        "source_photos/Osnos-JoeBiden.jpg",
        "source_photos/trump.jpg",
        "source_photos/obama-portrait.jpg"
    ]
    visualize_intersection(paths)