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
        if self.landmarker is None:
            options = self.vision.FaceLandmarkerOptions(
                base_options=self.BaseOptions(model_asset_path="face_landmarker.task"),
                running_mode=self.vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self.landmarker = self.vision.FaceLandmarker.create_from_options(options)

        h, w = image.shape[:2]

        # Save image temporarily (MediaPipe Image requires file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        cv2.imwrite(temp_path, image)

        try:
            from mediapipe.tasks.python.vision.core.image import Image
            mp_image = Image.create_from_file(temp_path)

            results = self.landmarker.detect(mp_image)

            if not results.face_landmarks:
                raise ValueError("No face detected in image")

            landmarks = results.face_landmarks[0]
            points = []
            for idx in self.MEDIAPIPE_TO_68:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    points.append((pt.x * w, pt.y * h))
                else:
                    # Fallback if index out of range
                    points.append((w / 2, h / 2))

            return np.array(points, dtype=np.float64)
        finally:
            # Clean up temp file
            import os
            os.unlink(temp_path)

    def __del__(self):
        """Cleanup - ignore errors during Python shutdown."""
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except (TypeError, AttributeError):
                # Ignore errors during interpreter shutdown
                pass


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
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        if len(faces) == 0:
            raise ValueError("No face detected in image")

        shape = self.predictor(gray, faces[0])
        points = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(68)],
            dtype=np.float64
        )
        return points


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
