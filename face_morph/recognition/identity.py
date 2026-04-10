"""Face identity matching and weight calculation."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2


@dataclass
class FaceAppearance:
    """One detection of a face in an image."""
    image_idx: int
    face_idx: int
    landmarks: np.ndarray
    embedding: Optional[np.ndarray] = None

    def __repr__(self):
        return f"FaceAppearance(img={self.image_idx}, face={self.face_idx})"


@dataclass
class FaceIdentity:
    """Represents a unique person across images."""
    person_id: int
    appearances: list[FaceAppearance] = field(default_factory=list)

    def __repr__(self):
        return f"FaceIdentity(id={self.person_id}, appearances={len(self.appearances)})"


class IdentityMatcher:
    """Matches faces across images using embeddings."""

    def __init__(self, threshold: float = 0.6):
        """Initialize identity matcher.

        Args:
            threshold: Maximum face distance for same identity (default 0.6).
                      Lower values are more strict.
        """
        self.threshold = threshold
        self._face_recognition = None

    def _ensure_face_recognition(self):
        """Lazy import face_recognition module."""
        if self._face_recognition is None:
            try:
                import face_recognition
                self._face_recognition = face_recognition
            except ImportError as e:
                raise ImportError(
                    "face_recognition package is required for group photo mode. "
                    "Install with: pip install face-recognition"
                ) from e
        return self._face_recognition

    def extract_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract 128-d face embedding using face_recognition.

        Args:
            image: BGR image (H, W, 3)
            landmarks: (68, 2) landmark array

        Returns:
            128-d face embedding vector
        """
        face_recognition = self._ensure_face_recognition()

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find face location from landmarks
        x_min = int(np.min(landmarks[:, 0]))
        x_max = int(np.max(landmarks[:, 0]))
        y_min = int(np.min(landmarks[:, 1]))
        y_max = int(np.max(landmarks[:, 1]))

        # Add some padding
        padding = int(0.2 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(rgb_image.shape[1], x_max + padding)
        y_max = min(rgb_image.shape[0], y_max + padding)

        face_location = (y_min, x_max, y_max, x_min)  # top, right, bottom, left

        # Get face encoding
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=[face_location],
            num_jitters=1
        )

        if not encodings:
            # Fallback: try without known location
            encodings = face_recognition.face_encodings(rgb_image, num_jitters=1)

        if not encodings:
            raise ValueError(f"Could not extract face embedding for face at ({x_min}, {y_min})")

        return encodings[0]

    def match_faces(
        self,
        appearances: list[FaceAppearance]
    ) -> list[FaceIdentity]:
        """Group face appearances by identity.

        Uses dynamic threshold calibration for single-image cases where
        all faces are guaranteed to be from different people.

        Args:
            appearances: List of all face appearances across images

        Returns:
            List of FaceIdentity objects, each containing appearances of the same person
        """
        face_recognition = self._ensure_face_recognition()

        if not appearances:
            return []

        # Extract embeddings
        embeddings = []
        valid_appearances = []

        for app in appearances:
            if app.embedding is not None:
                embeddings.append(app.embedding)
                valid_appearances.append(app)

        if not embeddings:
            return []

        embeddings = np.array(embeddings)

        # Check if all faces are from the same image
        image_indices = set(app.image_idx for app in appearances)
        single_image = len(image_indices) == 1

        if single_image:
            # For single image, we know all faces are unique
            # Calibrate threshold by binary search to find value that produces all unique identities
            print("  Calibrating threshold for single-image case...")
            calibrated_threshold = self._calibrate_threshold_for_unique_faces(embeddings, valid_appearances)
            print(f"  Calibrated threshold: {calibrated_threshold:.3f}")

            # Use the calibrated threshold for clustering
            identities = self._cluster_embeddings(embeddings, valid_appearances, threshold=calibrated_threshold)

            # Verify result
            if len(identities) == len(valid_appearances):
                print(f"  All {len(identities)} faces identified as unique people")
            else:
                print(f"  Warning: Only {len(identities)} identities for {len(valid_appearances)} faces")

            return identities
        else:
            # For multiple images, use the configured threshold
            identities = self._cluster_embeddings(embeddings, valid_appearances, threshold=self.threshold)
            return identities

    def _calibrate_threshold_for_unique_faces(
        self,
        embeddings: np.ndarray,
        appearances: list[FaceAppearance],
        min_threshold: float = 0.05,
        max_threshold: float = 0.8
    ) -> float:
        """Find threshold that makes all faces unique (for single-image case).

        Finds a threshold just below the minimum distance between any two
        different faces, ensuring all faces remain separate identities.

        Args:
            embeddings: Face embedding vectors
            appearances: Face appearances
            min_threshold: Minimum threshold to try
            max_threshold: Maximum threshold to try

        Returns:
            Calibrated threshold value (guaranteed to make all faces unique)
        """
        face_recognition = self._ensure_face_recognition()
        n_faces = len(embeddings)

        # If only one face, any threshold works
        if n_faces <= 1:
            return self.threshold

        # Find the minimum distance between any two different faces
        min_pair_distance = float('inf')
        closest_pair = (0, 0)
        all_distances = []

        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                distance = face_recognition.face_distance([embeddings[i]], embeddings[j])[0]
                all_distances.append((i, j, distance))
                if distance < min_pair_distance:
                    min_pair_distance = distance
                    closest_pair = (i, j)

        # Sort distances to see all pairs
        all_distances.sort(key=lambda x: x[2])

        print(f"    Closest face pairs (top 5):")
        for i, j, dist in all_distances[:5]:
            print(f"      Face {i} <-> Face {j}: {dist:.4f}")

        # For single image, set threshold to 99% of minimum distance
        # This ensures all faces remain unique
        safety_margin = 0.99
        calibrated = min_pair_distance * safety_margin

        # Clamp to reasonable bounds
        calibrated = max(calibrated, min_threshold)
        calibrated = min(calibrated, max_threshold)

        print(f"    Minimum distance: {min_pair_distance:.4f} (faces {closest_pair[0]} and {closest_pair[1]})")
        print(f"    Calibrated threshold: {calibrated:.4f}")

        return calibrated

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        appearances: list[FaceAppearance],
        threshold: float | None = None
    ) -> list[FaceIdentity]:
        """Cluster embeddings into identities using face distance.

        Uses a greedy approach: for each face, find if it matches any existing
        identity. If not, create a new identity.

        Args:
            embeddings: Face embedding vectors
            appearances: Corresponding face appearances
            threshold: Distance threshold for matching (uses self.threshold if None)

        Returns:
            List of FaceIdentity clusters
        """
        face_recognition = self._ensure_face_recognition()
        threshold = threshold if threshold is not None else self.threshold

        identities: list[FaceIdentity] = []

        for i, (embedding, appearance) in enumerate(zip(embeddings, appearances)):
            matched_identity = None

            # Check against all existing identities
            for identity in identities:
                # Compare with first appearance of this identity
                ref_embedding = embeddings[appearances.index(identity.appearances[0])]
                distance = face_recognition.face_distance([ref_embedding], embedding)[0]

                if distance < threshold:
                    matched_identity = identity
                    break

            if matched_identity is None:
                # Create new identity
                matched_identity = FaceIdentity(person_id=len(identities))
                identities.append(matched_identity)

            matched_identity.appearances.append(appearance)

        return identities

    def compute_weights(
        self,
        identities: list[FaceIdentity]
    ) -> dict[tuple[int, int], float]:
        """Compute weight for each face appearance.

        Each unique person gets equal total weight, split across their appearances.

        Args:
            identities: List of FaceIdentity objects

        Returns:
            Mapping from (image_idx, face_idx) to weight
        """
        if not identities:
            return {}

        num_unique_people = len(identities)
        person_weight = 1.0 / num_unique_people

        weights = {}
        for identity in identities:
            num_appearances = len(identity.appearances)
            appearance_weight = person_weight / num_appearances

            for appearance in identity.appearances:
                key = (appearance.image_idx, appearance.face_idx)
                weights[key] = appearance_weight

        return weights

    def generate_identity_report(
        self,
        identities: list[FaceIdentity],
        weights: dict[tuple[int, int], float]
    ) -> str:
        """Generate a text report of detected identities and weights.

        Args:
            identities: List of FaceIdentity objects
            weights: Weight mapping from compute_weights

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Detected {len(identities)} unique people")
        lines.append("=" * 60)

        for identity in identities:
            lines.append(f"\nPerson {identity.person_id + 1}:")
            lines.append(f"  Appearances: {len(identity.appearances)}")

            person_total_weight = sum(
                weights.get((app.image_idx, app.face_idx), 0)
                for app in identity.appearances
            )
            lines.append(f"  Total weight: {person_total_weight:.4f}")

            for app in identity.appearances:
                weight = weights.get((app.image_idx, app.face_idx), 0)
                lines.append(f"    - Image {app.image_idx}, Face {app.face_idx}: {weight:.4f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
