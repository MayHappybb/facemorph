"""Face clustering by identity using embedding similarity."""

import numpy as np
from typing import Optional


def cluster_faces_by_identity(
    embeddings: list[np.ndarray],
    threshold: float = 0.6,
    method: str = "greedy"
) -> list[int]:
    """Assign cluster labels to faces based on embedding similarity.

    Args:
        embeddings: List of face embedding vectors (each 128-d)
        threshold: Maximum distance for same identity
        method: Clustering algorithm ("greedy" or "complete")

    Returns:
        List of cluster labels (same label = same person)
    """
    if not embeddings:
        return []

    if method == "greedy":
        return _greedy_cluster(embeddings, threshold)
    elif method == "complete":
        return _complete_linkage_cluster(embeddings, threshold)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _face_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute Euclidean distance between two face embeddings."""
    return np.linalg.norm(embedding1 - embedding2)


def _greedy_cluster(embeddings: list[np.ndarray], threshold: float) -> list[int]:
    """Greedy clustering: assign each face to first matching cluster.

    This is efficient and works well when faces are clearly distinct.
    """
    if not embeddings:
        return []

    labels = []
    cluster_centers = []  # First face in each cluster

    for embedding in embeddings:
        matched_cluster = None

        # Check against all existing clusters
        for cluster_idx, center in enumerate(cluster_centers):
            distance = _face_distance(embedding, center)
            if distance < threshold:
                matched_cluster = cluster_idx
                break

        if matched_cluster is None:
            # Create new cluster
            matched_cluster = len(cluster_centers)
            cluster_centers.append(embedding)

        labels.append(matched_cluster)

    return labels


def _complete_linkage_cluster(embeddings: list[np.ndarray], threshold: float) -> list[int]:
    """Complete linkage clustering: more conservative matching.

    A face is assigned to a cluster only if it's within threshold of ALL
    members of that cluster. This prevents chaining errors.
    """
    if not embeddings:
        return []

    n = len(embeddings)
    # Initialize: each face in its own cluster
    clusters = [[i] for i in range(n)]

    # Compute all pairwise distances
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            distances[(i, j)] = _face_distance(embeddings[i], embeddings[j])
            distances[(j, i)] = distances[(i, j)]

    # Merge clusters
    changed = True
    while changed:
        changed = False
        new_clusters = []
        merged = set()

        for i, cluster_i in enumerate(clusters):
            if i in merged:
                continue

            merged.add(i)
            current_cluster = list(cluster_i)

            for j, cluster_j in enumerate(clusters[i + 1:], i + 1):
                if j in merged:
                    continue

                # Check if clusters can be merged (complete linkage)
                can_merge = True
                for idx_i in current_cluster:
                    for idx_j in cluster_j:
                        if distances.get((idx_i, idx_j), float('inf')) >= threshold:
                            can_merge = False
                            break
                    if not can_merge:
                        break

                if can_merge:
                    current_cluster.extend(cluster_j)
                    merged.add(j)
                    changed = True

            new_clusters.append(current_cluster)

        clusters = new_clusters

    # Assign labels
    labels = [0] * n
    for cluster_idx, cluster in enumerate(clusters):
        for face_idx in cluster:
            labels[face_idx] = cluster_idx

    return labels
