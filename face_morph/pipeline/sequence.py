"""Morph sequence generation."""

import os
import numpy as np
import cv2
from .morph import morph_faces


def generate_morph_sequence(
    image1: np.ndarray,
    image2: np.ndarray,
    landmarks1: np.ndarray,
    landmarks2: np.ndarray,
    num_frames: int = 30,
    output_dir: str = "morph_frames",
    warper: str = 'opencv'
) -> list[np.ndarray]:
    """Generate morph sequence from image1 to image2.

    Args:
        image1: First face image
        image2: Second face image
        landmarks1: (68, 2) landmarks for image1
        landmarks2: (68, 2) landmarks for image2
        num_frames: Number of frames to generate
        output_dir: Directory to save frames
        warper: 'opencv' or 'inverse'

    Returns:
        List of frames
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = []

    for k in range(num_frames):
        t = k / (num_frames - 1) if num_frames > 1 else 0.0

        print(f"  Frame {k + 1}/{num_frames} (alpha={t:.3f})")

        frame = morph_faces(
            image1, image2,
            landmarks1, landmarks2,
            alpha=t,
            warper=warper
        )

        frames.append(frame)
        cv2.imwrite(f"{output_dir}/frame_{k:04d}.png", frame)

    return frames


def save_video(
    frames_dir: str,
    output_path: str,
    fps: int = 30
):
    """Save frames as video using ffmpeg if available, otherwise skip.

    Args:
        frames_dir: Directory containing frame_%04d.png files
        output_path: Output video path
        fps: Frames per second
    """
    # Try ffmpeg first
    import subprocess
    import shutil

    if shutil.which("ffmpeg"):
        print(f"Creating video with ffmpeg: {output_path}")
        # ffmpeg command with pad filter to ensure even dimensions
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure even dimensions
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e}")
            print("Frames saved as PNG files instead")
    else:
        print("ffmpeg not found. Skipping video creation.")
        print(f"Individual frames saved in {frames_dir}")
