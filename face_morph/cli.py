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
    parser.add_argument("images", nargs="+", help="Paths to face images (2 or more)")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Comma-separated weights (e.g., '0.3,0.4,0.3'). Default: equal weights"
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
        help="Generate morph sequence (only for 2 images)"
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

    # Validate number of images
    if len(args.images) < 2:
        print("Error: Need at least 2 images", file=sys.stderr)
        sys.exit(1)

    # Validate sequence mode
    if args.sequence and len(args.images) != 2:
        print("Error: --sequence only supported for 2 images", file=sys.stderr)
        sys.exit(1)

    # Parse weights
    if args.weights:
        try:
            weights = [float(w) for w in args.weights.split(",")]
        except ValueError:
            print("Error: Invalid weights format. Use comma-separated numbers.", file=sys.stderr)
            sys.exit(1)

        if len(weights) != len(args.images):
            print(f"Error: {len(weights)} weights provided but {len(args.images)} images", file=sys.stderr)
            sys.exit(1)

        if any(w < 0 for w in weights):
            print("Error: Weights must be non-negative", file=sys.stderr)
            sys.exit(1)

        if sum(weights) == 0:
            print("Error: Sum of weights cannot be zero", file=sys.stderr)
            sys.exit(1)
    else:
        # Default: equal weights
        weights = [1.0 / len(args.images)] * len(args.images)

    # Load images
    images = []
    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image: {path}", file=sys.stderr)
            sys.exit(1)
        images.append(img)

    print(f"Loaded {len(images)} images:")
    for i, (path, img) in enumerate(zip(args.images, images)):
        print(f"  [{i}] {path} ({img.shape[1]}x{img.shape[0]})")

    print(f"Weights: {[round(w, 3) for w in weights]}")

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
    landmarks = []
    try:
        for i, img in enumerate(images):
            lm = detector.detect(img)
            landmarks.append(lm)
            print(f"  [{i}] {len(lm)} landmarks")
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
            images, landmarks,
            num_frames=args.num_frames,
            output_dir=output_dir,
            warper=args.warper
        )

        # Save video
        video_path = os.path.splitext(args.output)[0] + ".mp4"
        save_video(output_dir, video_path, fps=args.fps)
    else:
        print(f"\nMorphing {len(images)} faces...")
        print(f"Using {args.warper} warper...")
        result = morph_faces(
            images, landmarks, weights,
            warper=args.warper
        )

        # Save result
        success = cv2.imwrite(args.output, result)
        if not success:
            print(f"Error: Failed to save output to {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"Saved result to {args.output} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()