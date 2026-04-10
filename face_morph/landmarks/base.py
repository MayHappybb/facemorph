"""Landmark detection backends."""

from abc import ABC, abstractmethod
import numpy as np


class LandmarkDetector(ABC):
    """Abstract base class for landmark detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect facial landmarks in image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            (68, 2) array of landmark coordinates (x, y)

        Raises:
            ValueError: if no face detected
        """
        pass

    @abstractmethod
    def detect_all(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect all faces in image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            List of (68, 2) landmark arrays for each detected face

        Raises:
            ValueError: if no faces detected
        """
        pass


class MediaPipeDetector(LandmarkDetector):
    """MediaPipe Face Landmarker (468 points mapped to 68)."""

    # Mapping from MediaPipe 468 indices to standard 68-point model
    # Based on iBUG 300-W convention
    MEDIAPIPE_TO_68 = [
        # Jawline (0-16) - approximate from face contour
        234, 93, 132, 58, 172, 136, 150, 149, 148, 152, 377, 378, 379,
        365, 397, 288, 361, 323,
        # Left eyebrow (17-21)
        70, 63, 105, 66, 107,
        # Right eyebrow (22-26)
        336, 296, 334, 293, 300,
        # Nose bridge (27-30)
        168, 6, 197, 195, 5,
        # Nose base (31-35) - approximate
        48, 49, 4, 278, 279,
        # Left eye (36-41)
        33, 160, 158, 133, 153, 144,
        # Right eye (42-47)
        362, 385, 387, 263, 373, 380,
        # Outer mouth (48-59)
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61,
        # Inner mouth (60-67)
        78, 95, 88, 178, 87, 14, 317, 402,
    ]

    def __init__(self):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        self.vision = vision
        self.BaseOptions = BaseOptions
        self.landmarker = None

        # Download model if not present
        self._ensure_model()

    def _ensure_model(self):
        """Ensure the face landmarker model is available."""
        import os
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            # Download from Google's storage
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            import urllib.request
            print(f"Downloading MediaPipe face landmarker model...")
            urllib.request.urlretrieve(url, model_path)
            print(f"Model saved to {model_path}")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect landmarks using MediaPipe."""
        faces = self.detect_all(image)
        if not faces:
            raise ValueError("No face detected in image")
        return faces[0]

    def detect_all(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect all faces using MediaPipe with OpenCV face detection for region proposal."""
        if self.landmarker is None:
            options = self.vision.FaceLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path="face_landmarker.task"),
                running_mode=self.vision.RunningMode.IMAGE,
                num_faces=1,  # Process one face at a time from crops
            )
            self.landmarker = self.vision.FaceLandmarker.create_from_options(options)

        h, w = image.shape[:2]

        # First, use OpenCV Haar cascade to detect face regions
        # This works better for small faces in large images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        # Load Haar cascades
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Detect frontal faces with relaxed parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More granular scale steps
            minNeighbors=3,    # Lower threshold to catch more faces
            minSize=(30, 30)   # Smaller minimum face size
        )

        # Also detect profile faces
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        profile_faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # Combine all detections
        import numpy as np
        all_faces = []
        if len(faces) > 0:
            all_faces.extend(faces)
        if len(profile_faces) > 0:
            all_faces.extend(profile_faces)

        if len(all_faces) == 0:
            faces = np.array([])
        else:
            faces = np.array(all_faces)

        print(f"  OpenCV detected {len(faces)} faces (frontal + profile)")

        # Save debug image with OpenCV detections
        import os
        debug_dir = "debug_detection"
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = image.copy()
        for i, (x, y, fw, fh) in enumerate(faces):
            cv2.rectangle(debug_img, (x, y), (x+fw, y+fh), (0, 255, 0), 3)
            cv2.putText(debug_img, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_dir}/opencv_detections.jpg", debug_img)
        print(f"  Saved OpenCV detections to {debug_dir}/opencv_detections.jpg")

        if len(faces) == 0:
            return []

        # Remove duplicate/overlapping detections using NMS-like approach
        # High thresholds to only remove obvious duplicates (same face detected twice)
        faces = self._remove_duplicate_detections(faces, overlap_threshold=0.5, center_distance_threshold=0.3)
        print(f"  After deduplication: {len(faces)} faces")

        all_face_landmarks = []
        import tempfile
        from mediapipe.tasks.python.vision.core.image import Image

        for i, (x, y, fw, fh) in enumerate(faces):
            # Add padding around face for better landmark detection
            padding = int(0.3 * max(fw, fh))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + fw + padding)
            y2 = min(h, y + fh + padding)

            # Crop face region
            face_crop = image[y1:y2, x1:x2]

            # Save to temp file for MediaPipe
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                temp_path = f.name
            cv2.imwrite(temp_path, face_crop)

            try:
                mp_image = Image.create_from_file(temp_path)
                results = self.landmarker.detect(mp_image)

                if results.face_landmarks:
                    # Get landmarks and map back to original image coordinates
                    face_landmarks = results.face_landmarks[0]
                    points = []
                    for idx in self.MEDIAPIPE_TO_68:
                        if idx < len(face_landmarks):
                            pt = face_landmarks[idx]
                            # Map from crop coordinates to original image coordinates
                            orig_x = pt.x * (x2 - x1) + x1
                            orig_y = pt.y * (y2 - y1) + y1
                            points.append((orig_x, orig_y))
                        else:
                            points.append((w / 2, h / 2))
                    all_face_landmarks.append(np.array(points, dtype=np.float64))
            finally:
                import os
                os.unlink(temp_path)

        print(f"  MediaPipe landmarks for {len(all_face_landmarks)} faces")
        return all_face_landmarks

    def _remove_duplicate_detections(
        self,
        faces: np.ndarray,
        overlap_threshold: float = 0.3,
        center_distance_threshold: float = 0.2
    ) -> np.ndarray:
        """Remove overlapping face detections using IoU and center distance.

        Args:
            faces: Array of (x, y, w, h) detections
            overlap_threshold: IoU threshold for considering detections as duplicates
            center_distance_threshold: Max center distance as ratio of face size

        Returns:
            Filtered array of face detections
        """
        if len(faces) == 0:
            return faces

        # Calculate centers and sizes
        centers = np.column_stack([
            faces[:, 0] + faces[:, 2] / 2,  # cx = x + w/2
            faces[:, 1] + faces[:, 3] / 2   # cy = y + h/2
        ])
        sizes = np.maximum(faces[:, 2], faces[:, 3])  # max(w, h)

        # Sort by size (largest first) - prefer larger detections
        sorted_indices = np.argsort(sizes)[::-1]

        keep = []
        for i in sorted_indices:
            cx1, cy1 = centers[i]
            size1 = sizes[i]

            # Check distance to already kept faces
            is_duplicate = False
            for j in keep:
                cx2, cy2 = centers[j]
                size2 = sizes[j]

                # Calculate center distance normalized by face size
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                avg_size = (size1 + size2) / 2
                normalized_distance = distance / avg_size

                # Also check IoU
                x1, y1, w1, h1 = faces[i]
                x2, y2, w2, h2 = faces[j]
                area1, area2 = w1 * h1, w2 * h2

                xi = max(x1, x2)
                yi = max(y1, y2)
                wi = min(x1 + w1, x2 + w2) - xi
                hi = min(y1 + h1, y2 + h2) - yi

                iou = 0
                if wi > 0 and hi > 0:
                    intersection = wi * hi
                    union = area1 + area2 - intersection
                    iou = intersection / union

                # Mark as duplicate if centers are close OR IoU is high
                if normalized_distance < center_distance_threshold or iou > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(i)

        return faces[keep]

    def __del__(self):
        """Cleanup - ignore all errors during Python shutdown."""
        if getattr(self, 'landmarker', None) is not None:
            try:
                # Try to close, but ignore any errors during shutdown
                import sys
                if not sys.is_finalizing():
                    self.landmarker.close()
            except Exception:
                # Ignore all cleanup errors during interpreter shutdown
                pass
            finally:
                # Clear reference to prevent further cleanup attempts
                self.landmarker = None


class DlibDetector(LandmarkDetector):
    """dlib 68-point landmark detector."""

    def __init__(self, predictor_path: str):
        """Initialize dlib detector.

        Args:
            predictor_path: path to shape_predictor_68_face_landmarks.dat
        """
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect landmarks using dlib."""
        faces = self.detect_all(image)
        if not faces:
            raise ValueError("No face detected in image")
        return faces[0]

    def detect_all(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect all faces using dlib."""
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        if len(faces) == 0:
            return []

        all_faces = []
        for face in faces:
            shape = self.predictor(gray, face)
            points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)],
                dtype=np.float64
            )
            all_faces.append(points)

        return all_faces


def create_detector(backend: str, **kwargs) -> LandmarkDetector:
    """Factory function to create landmark detector.

    Args:
        backend: 'mediapipe' or 'dlib'
        **kwargs: backend-specific arguments (e.g., predictor_path for dlib)

    Returns:
        LandmarkDetector instance
    """
    if backend == 'mediapipe':
        return MediaPipeDetector()
    elif backend == 'dlib':
        predictor_path = kwargs.get('predictor_path')
        if predictor_path is None:
            raise ValueError("dlib backend requires predictor_path argument")
        return DlibDetector(predictor_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# Import cv2 here to avoid issues with module loading
import cv2
