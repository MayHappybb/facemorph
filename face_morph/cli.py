"""Command-line interface for face morphing."""

import argparse
import sys
import cv2
from .landmarks.base import create_detector
from .pipeline.morph import morph_faces
from .pipeline.sequence import generate_morph_sequence, save_video


def main():
    parser = argparse.ArgumentParser(
        description="Feature-based face morphing"
    )
    parser.add_argument("image1", help="Path to first face image")
    parser.add_argument("image2", help="Path to second face image")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Blending weight (0=pure image1, 1=pure image2, 0.5=average)"
    )
    parser.add_argument(
        "--backend", choices=["mediapipe", "dlib"], default="mediapipe",
        help="Landmark detection backend"
    )
    parser.add_argument(
        "--dlib-model", default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib shape predictor (dlib backend only)"
    )
    parser.add_argument(
        "--warper", choices=["opencv", "inverse"], default="opencv",
        help="Warping implementation (opencv=fast, inverse=pure python)"
    )
    parser.add_argument(
        "--sequence", action="store_true",
        help="Generate morph sequence instead of single frame"
    )
    parser.add_argument(
        "--num-frames", type=int, default=30,
        help="Number of frames for sequence"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--output", "-o", default="output.png",
        help="Output file path"
    )

    args = parser.parse_args()

    # Validate arguments
    if not (0.0 <= args.alpha <= 1.0):
        print("Error: alpha must be in [0, 1]", file=sys.stderr)
        sys.exit(1)

    # Load images
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)

    if img1 is None:
        print(f"Error: Could not load image: {args.image1}", file=sys.stderr)
        sys.exit(1)
    if img2 is None:
        print(f"Error: Could not load image: {args.image2}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded images: {args.image1} ({img1.shape}), {args.image2} ({img2.shape})")

    # Create landmark detector
    print(f"Using {args.backend} backend...")
    try:
        if args.backend == "dlib":
            detector = create_detector("dlib", predictor_path=args.dlib_model)
        else:
            detector = create_detector("mediapipe")
    except Exception as e:
        print(f"Error creating detector: {e}", file=sys.stderr)
        sys.exit(1)

    # Detect landmarks
    print("Detecting landmarks...")
    try:
        lm1 = detector.detect(img1)
        lm2 = detector.detect(img2)
        print(f"  Face 1: {len(lm1)} landmarks")
        print(f"  Face 2: {len(lm2)} landmarks")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Detection failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute
    if args.sequence:
        print(f"\nGenerating morph sequence ({args.num_frames} frames)...")
        import os
        output_dir = os.path.splitext(args.output)[0] + "_frames"
        frames = generate_morph_sequence(
            img1, img2, lm1, lm2,
            num_frames=args.num_frames,
            output_dir=output_dir,
            warper=args.warper
        )

        # Save video
        video_path = os.path.splitext(args.output)[0] + ".mp4"
        save_video(output_dir, video_path, fps=args.fps)
    else:
        print(f"\nMorphing with alpha={args.alpha}...")
        print(f"Using {args.warper} warper...")
        result = morph_faces(
            img1, img2, lm1, lm2,
            alpha=args.alpha,
            warper=args.warper
        )

        # Save result
        success = cv2.imwrite(args.output, result)
        if not success:
            print(f"Error: Failed to save output to {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
