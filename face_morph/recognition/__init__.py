"""Face recognition and identity matching module."""

from .identity import FaceIdentity, FaceAppearance, IdentityMatcher
from .clustering import cluster_faces_by_identity

__all__ = [
    'FaceIdentity',
    'FaceAppearance',
    'IdentityMatcher',
    'cluster_faces_by_identity',
]
