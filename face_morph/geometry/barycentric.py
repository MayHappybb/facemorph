"""Barycentric coordinate utilities."""

import numpy as np


def compute_barycentric_coords(
    point: np.ndarray,
    triangle: np.ndarray
) -> tuple[float, float, float]:
    """Compute barycentric coordinates of point w.r.t. triangle.

    Args:
        point: (2,) array [x, y]
        triangle: (3, 2) array of vertices [A, B, C]

    Returns:
        (alpha, beta, gamma) where point = alpha*A + beta*B + gamma*C
        and alpha + beta + gamma = 1

    Uses the area method:
        alpha = area(P, B, C) / area(A, B, C)
        beta  = area(A, P, C) / area(A, B, C)
        gamma = area(A, B, P) / area(A, B, C)
    """
    A, B, C = triangle
    P = point

    # Compute areas using cross product (2D analogue)
    def triangle_area(p1, p2, p3):
        return abs(np.cross(p2 - p1, p3 - p1)) / 2.0

    area_ABC = triangle_area(A, B, C)
    if area_ABC < 1e-10:  # Degenerate triangle
        return (-1, -1, -1)

    area_PBC = triangle_area(P, B, C)
    area_APC = triangle_area(A, P, C)
    area_ABP = triangle_area(A, B, P)

    alpha = area_PBC / area_ABC
    beta = area_APC / area_ABC
    gamma = area_ABP / area_ABC

    return (alpha, beta, gamma)


def point_in_triangle(
    point: np.ndarray,
    triangle: np.ndarray,
    eps: float = -1e-10
) -> bool:
    """Check if point is inside triangle (or on edge).

    Args:
        point: (2,) array [x, y]
        triangle: (3, 2) array of vertices
        eps: tolerance for numerical errors (negative allows slight boundary inclusion)

    Returns:
        True if point is inside or on edge of triangle
    """
    alpha, beta, gamma = compute_barycentric_coords(point, triangle)
    return (alpha >= eps) and (beta >= eps) and (gamma >= eps)


def barycentric_to_cartesian(
    bary_coords: tuple[float, float, float],
    triangle: np.ndarray
) -> np.ndarray:
    """Convert barycentric coordinates to Cartesian coordinates.

    Args:
        bary_coords: (alpha, beta, gamma)
        triangle: (3, 2) array of vertices [A, B, C]

    Returns:
        (2,) array [x, y]
    """
    alpha, beta, gamma = bary_coords
    A, B, C = triangle
    return alpha * A + beta * B + gamma * C
