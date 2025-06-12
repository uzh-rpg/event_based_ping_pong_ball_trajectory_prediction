import numpy as np
import itertools
import math
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, QhullError


def euclidean_distance(p1, p2):
    # Calculate Euclidean distance between two points.
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_farthest_triplet(points):
    # Find the triplet of points with the maximum total pairwise distance.
    max_distance = 0
    best_triplet = None

    try:
        hull = ConvexHull(points)
        border_points = points[hull.vertices]
    except QhullError:
        return None, None

    combinations = list(itertools.combinations(border_points, 3))
    for triplet in combinations:
        # Calculate the total pairwise distance
        d = euclidean_distance(triplet[0], triplet[1]) + euclidean_distance(triplet[0], triplet[2]) + euclidean_distance(triplet[1], triplet[2])

        # Update if this triplet has a greater total distance
        if d > max_distance:
            max_distance = d
            best_triplet = triplet

    return best_triplet, max_distance


def fit_ellipse(points):
    # Find the mean (center of the ellipse)
    center = points.mean(axis=0)

    # Apply PCA to find the orientation and axis lengths
    pca = PCA(n_components=2)
    pca.fit(points)
    radii = np.sqrt(pca.explained_variance_)
    orientation = pca.components_

    return center, radii, orientation


def fit_circle(points):
    # Fit a circle to three points.
    p1, p2, p3 = points
    
    # Compute the perpendicular bisectors of two line segments
    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2
    slope1 = -(p2[0] - p1[0]) / (p2[1] - p1[1]) if p2[1] != p1[1] else float("inf")
    slope2 = -(p3[0] - p2[0]) / (p3[1] - p2[1]) if p3[1] != p2[1] else float("inf")

    if slope1 == slope2:  # Collinear points can't define a unique circle
        return None

    # Solve for the intersection of the bisectors (circle center)
    if slope1 == float("inf"):
        xc = mid1[0]
        yc = slope2 * (xc - mid2[0]) + mid2[1]
    elif slope2 == float("inf"):
        xc = mid2[0]
        yc = slope1 * (xc - mid1[0]) + mid1[1]
    else:
        A = np.array([[-slope1, 1], [-slope2, 1]])
        b = np.array([mid1[1] - slope1 * mid1[0], mid2[1] - slope2 * mid2[0]])
        xc, yc = np.linalg.solve(A, b)

    radius = np.linalg.norm(np.array([xc, yc]) - p1)
    return (xc, yc, radius)


def ransac_circle_estimation(points, num_iterations=1000):
    # Find the best circle using RANSAC, maximizing inliers and density.
    best_circle = None
    max_score = 0

    for _ in range(num_iterations):
        sample, max_dist = find_farthest_triplet(points)

        if (sample is None) or (max_dist is None):
            return None, None, None

        # Fit a circle to the sampled points
        circle = fit_circle(sample)
        if circle is None:
            continue

        xc, yc, radius = circle

        # Compute the inlier points (inside or on the circle)
        distances = np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)
        inliers = points[distances <= radius]

        # Calculate the area of the circle
        circle_area = math.pi * radius**2

        # Calculate the density score
        if circle_area > 0:
            len(inliers) / circle_area
        else:
            pass

        score = len(inliers)

        if score > max_score:
            max_score = score
            best_circle = (xc, yc, radius)

    return best_circle

