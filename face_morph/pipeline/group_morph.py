"""Group photo face morphing pipeline with identity-aware weighting."""

import numpy as np
import cv2
from typing import Optional
from ..landmarks.base import LandmarkDetector
from ..recognition.identity import IdentityMatcher, FaceAppearance
from .morph import morph_faces


def detect_all_faces_in_images(
    images: list[np.ndarray],
    detector: LandmarkDetector
) -> tuple[list[list[np.ndarray]], list[tuple[int, int]]]:
    """Detect all faces in all images.

    Args:
        images: List of input images
        detector: Landmark detector with detect_all support

    Returns:
        Tuple of (all_landmarks_per_image, detection_counts)
        where all_landmarks_per_image[i] is list of landmark arrays for image i
        and detection_counts[i] is (detected, expected) for image i
    """
    all_landmarks = []
    detection_counts = []

    for i, image in enumerate(images):
        try:
            landmarks_list = detector.detect_all(image)
            count = len(landmarks_list)
            all_landmarks.append(landmarks_list)
            detection_counts.append((count, count))
        except Exception as e:
            # No faces detected
            all_landmarks.append([])
            detection_counts.append((0, 0))
            print(f"Warning: No faces detected in image {i}: {e}")

    return all_landmarks, detection_counts


def extract_face_embeddings(
    images: list[np.ndarray],
    all_landmarks: list[list[np.ndarray]],
    identity_matcher: IdentityMatcher
) -> list[list[Optional[np.ndarray]]]:
    """Extract face embeddings for all detected faces.

    Args:
        images: List of input images
        all_landmarks: Landmarks for each face in each image
        identity_matcher: Identity matcher for embedding extraction

    Returns:
        Nested list of embeddings: embeddings[i][j] is embedding for face j in image i
    """
    all_embeddings = []

    for img_idx, (image, landmarks_list) in enumerate(zip(images, all_landmarks)):
        image_embeddings = []
        for face_idx, landmarks in enumerate(landmarks_list):
            try:
                embedding = identity_matcher.extract_embedding(image, landmarks)
                image_embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Failed to extract embedding for image {img_idx}, face {face_idx}: {e}")
                image_embeddings.append(None)
        all_embeddings.append(image_embeddings)

    return all_embeddings


def create_face_appearances(
    all_landmarks: list[list[np.ndarray]],
    all_embeddings: list[list[Optional[np.ndarray]]]
) -> list[FaceAppearance]:
    """Create FaceAppearance objects for all valid faces.

    Args:
        all_landmarks: Nested list of landmarks
        all_embeddings: Nested list of embeddings

    Returns:
        List of FaceAppearance objects with embeddings
    """
    appearances = []

    for img_idx, (landmarks_list, embeddings_list) in enumerate(zip(all_landmarks, all_embeddings)):
        for face_idx, (landmarks, embedding) in enumerate(zip(landmarks_list, embeddings_list)):
            if embedding is not None:
                appearance = FaceAppearance(
                    image_idx=img_idx,
                    face_idx=face_idx,
                    landmarks=landmarks,
                    embedding=embedding
                )
                appearances.append(appearance)

    return appearances


def remove_duplicate_faces_by_landmarks(
    all_landmarks: list[list[np.ndarray]],
    similarity_threshold: float = 0.95
) -> list[list[np.ndarray]]:
    """Remove duplicate face detections by comparing landmarks.

    When OpenCV detects the same person multiple times, the landmarks
    will be very similar. Keep only one detection per person.

    Args:
        all_landmarks: Nested list of landmarks for each face in each image
        similarity_threshold: Minimum similarity to consider as same person (0-1)

    Returns:
        Filtered landmarks with duplicates removed
    """
    def landmarks_similarity(lm1: np.ndarray, lm2: np.ndarray) -> float:
        """Calculate similarity between two landmark sets.

        For detecting duplicate OpenCV detections, we compare both:
        1. Position (centroid distance) - same person should be near same location
        2. Shape (normalized correlation) - same person should have same face shape
        """
        # 1. Position similarity (centroid distance)
        c1 = np.mean(lm1, axis=0)
        c2 = np.mean(lm2, axis=0)
        position_distance = np.linalg.norm(c1 - c2)
        # Normalize by average face size (approx 100 pixels)
        position_similarity = max(0, 1 - position_distance / 100)

        # 2. Shape similarity (normalized correlation)
        centered1 = lm1 - c1
        centered2 = lm2 - c2
        norm1 = np.linalg.norm(centered1)
        norm2 = np.linalg.norm(centered2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        unit1 = centered1 / norm1
        unit2 = centered2 / norm2
        shape_correlation = np.sum(unit1 * unit2)
        shape_similarity = (shape_correlation + 1) / 2

        # Combined: both position AND shape must match for duplicates
        # Position is weighted more because OpenCV duplicates are offset
        combined = 0.7 * position_similarity + 0.3 * shape_similarity

        return combined

    filtered_landmarks = []

    for img_idx, landmarks_list in enumerate(all_landmarks):
        if not landmarks_list:
            filtered_landmarks.append([])
            continue

        keep_indices = []

        for i, lm1 in enumerate(landmarks_list):
            is_duplicate = False

            for j in keep_indices:
                lm2 = landmarks_list[j]
                similarity = landmarks_similarity(lm1, lm2)

                if similarity > similarity_threshold:
                    print(f"    Image {img_idx}: Face {i} is duplicate of face {j} (similarity: {similarity:.3f})")
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_indices.append(i)

        filtered = [landmarks_list[i] for i in keep_indices]
        filtered_landmarks.append(filtered)

        if len(keep_indices) < len(landmarks_list):
            print(f"  Image {img_idx}: Removed {len(landmarks_list) - len(keep_indices)} duplicates, kept {len(keep_indices)} faces")

    return filtered_landmarks


def save_detected_faces(
    images: list[np.ndarray],
    all_landmarks: list[list[np.ndarray]],
    output_dir: str = "detected_faces"
) -> None:
    """Save cropped face regions for inspection.

    Args:
        images: List of input images
        all_landmarks: Landmarks for each face in each image
        output_dir: Directory to save face crops
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for img_idx, (image, landmarks_list) in enumerate(zip(images, all_landmarks)):
        h, w = image.shape[:2]

        for face_idx, landmarks in enumerate(landmarks_list):
            # Calculate bounding box from landmarks
            x_min = int(max(0, np.min(landmarks[:, 0]) - 20))
            y_min = int(max(0, np.min(landmarks[:, 1]) - 20))
            x_max = int(min(w, np.max(landmarks[:, 0]) + 20))
            y_max = int(min(h, np.max(landmarks[:, 1]) + 20))

            # Crop face
            face_crop = image[y_min:y_max, x_min:x_max]

            # Save
            filename = f"{output_dir}/face_img{img_idx}_face{face_idx:02d}.jpg"
            cv2.imwrite(filename, face_crop)

    print(f"  Saved {sum(len(l) for l in all_landmarks)} face crops to {output_dir}/")


def morph_group_photos(
    images: list[np.ndarray],
    detector: LandmarkDetector,
    identity_matcher: IdentityMatcher,
    warper: str = 'opencv',
    show_report: bool = False
) -> tuple[np.ndarray, Optional[str]]:
    """Morph faces from group photos with identity-aware weighting.

    Args:
        images: List of group photos
        detector: Landmark detector (must support detect_all)
        identity_matcher: Identity matching engine
        warper: Warping implementation ('opencv' or 'inverse')
        show_report: Whether to generate identity report

    Returns:
        Tuple of (morphed_image, report_string or None)

    Raises:
        ValueError: If no faces detected or no valid embeddings
    """
    print(f"\nProcessing {len(images)} group photos...")

    # Step 1: Detect all faces in all images
    print("\nStep 1: Detecting faces...")
    all_landmarks, detection_counts = detect_all_faces_in_images(images, detector)

    total_faces = sum(len(landmarks) for landmarks in all_landmarks)
    if total_faces == 0:
        raise ValueError("No faces detected in any image")

    print(f"Detected {total_faces} faces across {len(images)} images:")
    for i, (detected, _) in enumerate(detection_counts):
        print(f"  Image {i}: {detected} faces")

    # Step 1.5: Remove duplicates by landmark similarity
    print("\nStep 1.5: Removing duplicate detections by landmark similarity...")
    all_landmarks = remove_duplicate_faces_by_landmarks(all_landmarks, similarity_threshold=0.95)

    total_faces_after = sum(len(landmarks) for landmarks in all_landmarks)
    if total_faces_after < total_faces:
        print(f"After deduplication: {total_faces_after} unique faces")

    # Step 2: Extract embeddings for each face
    print("\nStep 2: Extracting face embeddings...")
    all_embeddings = extract_face_embeddings(images, all_landmarks, identity_matcher)

    # Save detected faces for inspection
    save_detected_faces(images, all_landmarks, output_dir="detected_faces")

    # Step 3: Create face appearances
    appearances = create_face_appearances(all_landmarks, all_embeddings)

    if not appearances:
        raise ValueError("Could not extract embeddings for any faces")

    print(f"Successfully extracted embeddings for {len(appearances)} faces")

    # Step 4: Match faces by identity
    print("\nStep 3: Matching faces by identity...")
    identities = identity_matcher.match_faces(appearances)
    print(f"Identified {len(identities)} unique people")

    # Step 5: Compute weights
    print("\nStep 4: Computing weights...")
    weights_map = identity_matcher.compute_weights(identities)

    # Step 6: Prepare for morphing
    # Flatten all faces and their weights
    flat_images = []
    flat_landmarks = []
    flat_weights = []

    for img_idx, landmarks_list in enumerate(all_landmarks):
        for face_idx, landmarks in enumerate(landmarks_list):
            key = (img_idx, face_idx)
            if key in weights_map:
                flat_images.append(images[img_idx])
                flat_landmarks.append(landmarks)
                flat_weights.append(weights_map[key])

    if not flat_images:
        raise ValueError("No valid faces with weights to morph")

    # Normalize weights to sum to 1
    total_weight = sum(flat_weights)
    flat_weights = [w / total_weight for w in flat_weights]

    print(f"\nMorphing {len(flat_images)} faces...")
    for i, (img_idx, face_idx) in enumerate(weights_map.keys()):
        print(f"  Face {i+1}: Image {img_idx}, Face {face_idx}, Weight {flat_weights[i]:.4f}")

    # Step 7: Morph all faces
    print(f"\nStep 5: Morphing using {warper} warper...")
    result = morph_faces(flat_images, flat_landmarks, flat_weights, warper=warper)

    # Step 8: Generate report if requested
    report = None
    if show_report:
        report = identity_matcher.generate_identity_report(identities, weights_map)

    return result, report
