# Group Photo Face Morphing - Process Flow Documentation

## Overview

This document describes the complete workflow for processing group photos with automatic face detection, deduplication, and identity-aware morphing.

## System Architecture

```
Input Image(s)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: FACE REGION DETECTION (OpenCV Haar Cascade)           │
│  ├─ Detect frontal faces: haarcascade_frontalface_default.xml   │
│  ├─ Detect profile faces: haarcascade_profileface.xml           │
│  ├─ Merge and deduplicate overlapping detections                │
│  └─ Output: List of face bounding boxes (x, y, w, h)            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: FACIAL LANDMARK EXTRACTION (MediaPipe)                │
│  ├─ For each detected face region:                              │
│  │   ├─ Add 30% padding around region                           │
│  │   ├─ Crop face from original image                           │
│  │   ├─ Extract 468 landmarks using MediaPipe Face Landmarker   │
│  │   ├─ Map 468 → 68 landmarks (standard iBUG format)           │
│  │   └─ Transform coordinates back to original image space      │
│  └─ Output: List of (68, 2) landmark arrays per face            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: DUPLICATE DETECTION & REMOVAL (Landmark-Based)        │
│  ├─ Calculate position similarity (centroid distance)           │
│  ├─ Calculate shape similarity (normalized correlation)         │
│  ├─ Combined similarity = 0.7*position + 0.3*shape              │
│  ├─ If similarity > 0.95: mark as duplicate                     │
│  └─ Output: Filtered list of unique face landmarks              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: FACE EMBEDDING EXTRACTION (face_recognition)          │
│  ├─ For each unique face:                                       │
│  │   ├─ Calculate bounding box from landmarks                   │
│  │   ├─ Add 20% padding                                         │
│  │   ├─ Extract 128-d face embedding (dlib model)               │
│  │   └─ Store embedding with face metadata                      │
│  └─ Output: List of FaceAppearance objects                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: IDENTITY MATCHING & CLUSTERING                        │
│  ├─ Single Image Mode:                                          │
│  │   ├─ All faces treated as unique (different people)          │
│  │   └─ Skip clustering, assign each face unique identity       │
│  ├─ Multiple Images Mode:                                       │
│  │   ├─ Calculate pairwise face embedding distances             │
│  │   ├─ Calibrate threshold (99% of minimum distance)           │
│  │   ├─ Greedy clustering: distance < threshold = same person   │
│  │   └─ Group appearances by identity                           │
│  └─ Output: List of FaceIdentity objects                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 6: WEIGHT CALCULATION                                    │
│  ├─ Each unique person gets equal total weight: 1/N             │
│  ├─ Multiple appearances share weight equally:                  │
│  │   Person weight = 1/N                                        │
│  │   Appearance weight = (1/N) / (number of appearances)        │
│  └─ Output: Weight mapping (image_idx, face_idx) → weight       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 7: FACE MORPHING                                         │
│  ├─ For each face:                                              │
│  │   ├─ Calculate face center and scale                         │
│  │   ├─ Normalize landmarks to face-centered coordinates         │
│  │   ├─ Compute weighted mean shape across all faces            │
│  │   ├─ Triangulate mean shape (Delaunay)                       │
│  │   ├─ Warp face to mean shape (OpenCV piecewise affine)       │
│  │   └─ Store warped face                                       │
│  ├─ Blend all warped faces using computed weights               │
│  └─ Output: Final morphed image                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Output Image
```

## Detailed Stage Descriptions

### Stage 1: Face Region Detection

**Input:** Original group photo image(s)

**Process:**
1. Convert image to grayscale
2. Run `haarcascade_frontalface_default.xml` with parameters:
   - `scaleFactor=1.05` (fine-grained scale search)
   - `minNeighbors=3` (lower threshold for more detections)
   - `minSize=(30, 30)` (catch smaller faces)
3. Run `haarcascade_profileface.xml` with same parameters
4. Merge all detections
5. Apply Non-Maximum Suppression (NMS):
   - Remove detections with IoU > 0.5
   - Remove detections with center distance < 0.3 * avg_size

**Output:** List of bounding boxes `(x, y, width, height)` for each detected face

**Debug:** Saves `debug_detection/opencv_detections.jpg` with all boxes drawn

---

### Stage 2: Facial Landmark Extraction

**Input:** Face bounding boxes from Stage 1

**Process:**
For each bounding box:
1. Add 30% padding to include more context
2. Crop the face region from original image
3. Save to temporary file (MediaPipe requires file input)
4. Load `face_landmarker.task` model (MediaPipe)
5. Extract 468 facial landmarks
6. Map to 68-point format using predefined index mapping
7. Transform coordinates back to original image space
8. Clean up temporary file

**Mapping (468 → 68):**
- Jawline: 0-16
- Eyebrows: 17-26
- Nose: 27-35
- Eyes: 36-47
- Mouth: 48-67

**Output:** List of `(68, 2)` landmark arrays per face

---

### Stage 3: Duplicate Detection & Removal

**Input:** List of landmark arrays

**Process:**
For each pair of faces in same image:
1. Calculate position similarity:
   ```
   centroid_distance = ||mean(lm1) - mean(lm2)||
   position_sim = max(0, 1 - centroid_distance / 100)
   ```

2. Calculate shape similarity:
   ```
   centered1 = lm1 - mean(lm1)
   centered2 = lm2 - mean(lm2)
   unit1 = centered1 / ||centered1||
   unit2 = centered2 / ||centered2||
   correlation = sum(unit1 * unit2)
   shape_sim = (correlation + 1) / 2
   ```

3. Combined similarity:
   ```
   similarity = 0.7 * position_sim + 0.3 * shape_sim
   ```

4. If `similarity > 0.95`: mark as duplicate, keep only first

**Rationale:** OpenCV Haar often detects same face multiple times with slight offsets. Position is weighted 70% because duplicates are spatially close but may have different scales.

**Output:** Filtered list of unique landmark arrays

---

### Stage 4: Face Embedding Extraction

**Input:** Unique face landmarks

**Process:**
For each face:
1. Calculate bounding box from landmarks (min/max x, y)
2. Add 20% padding for face_recognition
3. Extract 128-dimensional embedding using dlib's face recognition model
4. Store in `FaceAppearance` object:
   - `image_idx`: which image
   - `face_idx`: face index in image
   - `landmarks`: (68, 2) array
   - `embedding`: (128,) vector

**Model:** ResNet-based (dlib `face_recognition_model_v1`)

**Output:** List of `FaceAppearance` objects

**Debug:** Saves cropped face images to `detected_faces/`

---

### Stage 5: Identity Matching

**Single Image Mode:**
- All faces are from different people by definition
- Skip clustering, assign each face unique identity
- Each `FaceIdentity` contains one `FaceAppearance`

**Multiple Images Mode:**
1. Calculate pairwise face embedding distances:
   ```
   distance = ||embedding_i - embedding_j||
   ```

2. Calibrate threshold:
   ```
   min_distance = minimum of all pairwise distances
   threshold = min_distance * 0.99  (clamped to [0.05, 0.8])
   ```

3. Greedy clustering:
   ```
   For each face:
       For each existing identity:
           If distance(face, identity_first) < threshold:
               Add to this identity
               Break
       If no match:
           Create new identity
   ```

**Output:** List of `FaceIdentity` objects (each person = one identity)

---

### Stage 6: Weight Calculation

**Formula:**
```
N = number of unique people
person_weight = 1 / N

For each identity:
    appearances = number of times this person appears
    For each appearance:
        weight = person_weight / appearances
```

**Example:**
- Photo 1: 10 people (A-J)
- Photo 2: 8 people (A, K-Q) - Person A appears twice
- Total unique: 17 people
- Person A: 1/17 total weight → 1/34 per appearance
- Others: 1/17 each

**Output:** Dictionary mapping `(image_idx, face_idx) → weight`

---

### Stage 7: Face Morphing

**Process:**

1. **Normalize Coordinates:**
   - For each face: calculate centroid and scale
   - Normalize landmarks: `(landmarks - centroid) / scale`

2. **Compute Mean Shape:**
   - Weighted average of all normalized landmark sets
   - Weights from Stage 6

3. **Determine Output Size:**
   - Calculate intersection of all face-centered bounding boxes
   - Ensures all faces fit without cropping

4. **Triangulation:**
   - Apply Delaunay triangulation on mean shape + frame points
   - Frame points ensure coverage of output canvas

5. **Warping (per face):**
   - For each triangle in triangulation:
     - Compute affine transform: source → destination
     - Apply warpAffine to triangle region
   - Methods:
     - `opencv`: Fast, uses OpenCV's optimized functions
     - `inverse`: Educational, pure Python inverse mapping

6. **Blending:**
   - Weighted sum of all warped faces
   - `result = Σ(warped_face_i × weight_i)`

**Output:** Final morphed image as numpy array

---

## Data Structures

### FaceAppearance
```python
@dataclass
class FaceAppearance:
    image_idx: int          # Which input image
    face_idx: int           # Face index within image
    landmarks: np.ndarray   # (68, 2) landmark coordinates
    embedding: np.ndarray   # (128,) face embedding
```

### FaceIdentity
```python
@dataclass
class FaceIdentity:
    person_id: int                  # Unique person identifier
    appearances: list[FaceAppearance]  # All instances of this person
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `minNeighbors` | 3 | OpenCV detection threshold (lower = more faces) |
| `scaleFactor` | 1.05 | OpenCV scale step (lower = finer search) |
| `minSize` | (30, 30) | Minimum face size in pixels |
| `padding` | 30% | Extra space around face crops |
| `similarity_threshold` | 0.95 | Landmark dedup threshold |
| `identity_threshold` | 0.6 | Face recognition threshold (multi-image) |

---

## Command Line Usage

```bash
# Single group photo
python morph.py group.jpg --group-photos -o average.png

# Multiple group photos with identity matching
python morph.py party1.jpg party2.jpg --group-photos -o result.png

# Show identity report
python morph.py group.jpg --group-photos --show-identities

# Adjust identity matching strictness
python morph.py *.jpg --group-photos --identity-threshold 0.5
```

---

## File Outputs

| Path | Description |
|------|-------------|
| `average.png` | Final morphed image |
| `detected_faces/` | Cropped face regions for inspection |
| `debug_detection/opencv_detections.jpg` | OpenCV detection visualization |

---

## Known Limitations

1. **Face Size:** Very small faces (< 30px) may not be detected
2. **Occlusion:** Partially occluded faces may be missed
3. **Profile Views:** Side-facing profiles have lower detection rate
4. **Lighting:** Extreme lighting can affect embedding accuracy
5. **Similar People:** Twins or very similar-looking people may be grouped

---

## Performance Notes

- **OpenCV Detection:** ~100-500ms per image
- **MediaPipe Landmarks:** ~50-100ms per face
- **Embedding Extraction:** ~20-50ms per face
- **Morphing:** ~100-300ms per face (OpenCV warper)
- **Total:** ~2-5 seconds for 15-face group photo

---

## Dependencies

- `opencv-python` >= 4.5.0 (face detection, warping)
- `mediapipe` >= 0.8.0 (landmark extraction)
- `face-recognition` >= 1.3.0 (embeddings, requires dlib)
- `numpy` >= 1.20.0 (array operations)
- `scipy` >= 1.7.0 (Delaunay triangulation)
