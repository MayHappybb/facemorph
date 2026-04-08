# Face Morphing Program

A feature-based face morphing implementation following the tutorial. Supports both OpenCV-optimized and pure Python inverse mapping warpers.

## Features

- **Two landmark backends**: MediaPipe (478→68 points) and dlib (68 points)
- **Two warping engines**:
  - `opencv`: Fast, optimized implementation using `cv2.warpAffine` (~100-300ms/frame)
  - `inverse`: Pure Python educational implementation showing the exact algorithm
- **Static averaging** and **animated morph sequences**
- **Modular architecture** with clean separation of concerns

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Frame (Average Face)

```bash
# Using MediaPipe (default, no model files needed)
python morph.py face_A.jpg face_B.jpg --output average.png

# Custom blend (alpha=0.7)
python morph.py face_A.jpg face_B.jpg --alpha 0.7 --output blend.png

# Using pure inverse mapping warper (slower but educational)
python morph.py face_A.jpg face_B.jpg --warper inverse --output average.png

# Using dlib (requires shape_predictor_68_face_landmarks.dat)
python morph.py face_A.jpg face_B.jpg --backend dlib --dlib-model path/to/model.dat
```

### Morph Sequence

```bash
# Generate 30-frame morph animation
python morph.py face_A.jpg face_B.jpg --sequence --num-frames 30 --output morph.mp4

# Custom frame rate
python morph.py face_A.jpg face_B.jpg --sequence --num-frames 60 --fps 30
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `image1`, `image2` | Input face images | (required) |
| `--alpha` | Blending weight [0-1] | 0.5 |
| `--backend` | Landmark detector: `mediapipe` or `dlib` | mediapipe |
| `--warper` | Warping engine: `opencv` or `inverse` | opencv |
| `--sequence` | Generate morph sequence | False |
| `--num-frames` | Number of frames for sequence | 30 |
| `--output` | Output file path | output.png |

## Project Structure

```
face_morph/
├── landmarks/      # Landmark detection backends
├── geometry/       # Affine transforms, barycentric coords, Delaunay
├── warping/        # OpenCV and inverse mapping warpers
├── blending/       # Alpha blending
└── pipeline/       # Full morph pipeline

morph.py            # CLI entry point
```

## Algorithm Pipeline

1. **Detect landmarks** (68 points) on both faces
2. **Compute mean shape** by averaging corresponding landmarks
3. **Delaunay triangulation** on mean shape
4. **Piecewise affine warp** both faces to mean shape
5. **Alpha blend** the warped images

See the `face_morphing_tutorial.tex` for detailed mathematical derivations.
