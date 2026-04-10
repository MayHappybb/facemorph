# FaceMorph

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A feature-based face morphing implementation with dual warping engines and normalized coordinate system for handling images of different sizes without distortion.

![Morph Example](https://user-images.githubusercontent.com/placeholder/morph.gif)

## Features

- **Dual Landmark Backends**: MediaPipe (468 points) and dlib (68 points)
- **Two Warping Engines**:
  - `opencv`: Fast optimized implementation (~100-300ms/frame)
  - `inverse`: Pure Python educational implementation showing exact algorithm
- **Normalized Coordinates**: Handles different image sizes without stretching
- **Morph Sequences**: Generate smooth frame-by-frame animations
- **Group Photo Support**: Average faces from group photos with automatic identity deduplication
- **Comprehensive Tutorial**: Full LaTeX documentation with mathematical derivations

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/facemorph.git
cd facemorph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Single frame average
python morph.py face1.jpg face2.jpg --output average.png

# Morph sequence (30fps, 60 frames = 2 seconds)
python morph.py face1.jpg face2.jpg --sequence --num-frames 60 --fps 30
```

## Algorithm Pipeline

1. **Detect Landmarks** → 68 facial feature points
2. **Normalize** → Centered coordinates (height=1, aspect preserved)
3. **Triangulate** → Delaunay mesh on mean shape
4. **Warp** → Piecewise affine transformation to common geometry
5. **Blend** → Alpha compositing

## Project Structure

```
face_morph/
├── landmarks/      # MediaPipe & dlib backends
├── geometry/       # Affine, barycentric, Delaunay
├── warping/        # OpenCV & inverse mapping engines
├── blending/       # Alpha compositing
└── pipeline/       # Full morph pipeline

tutorial/           # Comprehensive LaTeX tutorial
```

## Documentation

See [`tutorial/face_morphing_tutorial.pdf`](tutorial/face_morphing_tutorial.pdf) for the complete mathematical derivation covering:
- Facial landmark detection
- Affine transformations and homogeneous coordinates
- Barycentric coordinates and point-in-triangle tests
- Delaunay triangulation
- Piecewise affine warping with inverse mapping
- Alpha blending and morph sequence generation

## Usage

```bash
# Average two faces (α=0.5)
python morph.py face_a.jpg face_b.jpg --alpha 0.5

# Custom blend (70% face B)
python morph.py face_a.jpg face_b.jpg --alpha 0.7

# Use pure inverse mapping (slower, educational)
python morph.py face_a.jpg face_b.jpg --warper inverse

# Full morph animation
python morph.py face_a.jpg face_b.jpg --sequence --num-frames 60
```

## Group Photo Mode

Average faces from group photos with automatic identity deduplication. When the same person appears in multiple photos, their total weight is split across appearances.

```bash
# Single group photo (equal weights for all faces)
python morph.py group_photo.jpg --group-photos -o average.png

# Multiple group photos with identity matching
python morph.py party1.jpg party2.jpg --group-photos -o party_average.png

# Show identity report with weights
python morph.py group1.jpg group2.jpg --group-photos --show-identities

# Adjust identity matching threshold (lower = more strict)
python morph.py party*.jpg --group-photos --identity-threshold 0.5
```

**Example weight calculation:**
- Photo 1 has 10 people (A-J)
- Photo 2 has 8 people (A, K-Q) - Person A appears in both
- Result: 5 unique people, Person A's two appearances share 1/5 weight (1/10 each)
- Others get 1/5 weight each

**Requirements:** Group photo mode requires `face-recognition` package (included in requirements.txt).

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

Based on the classic feature-based morphing algorithm by Beier and Neely (SIGGRAPH 1992).
