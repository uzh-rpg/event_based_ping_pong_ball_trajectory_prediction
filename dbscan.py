import cv2
import copy
import math
import numpy as np
import dv_processing as dv
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from circle_fitting import *
from circle_fit import taubinSVD
from scipy.spatial import ConvexHull, QhullError


def plot_dbscan_result_cv(X_scaled, clusters, t_upper_limit, events, circularities):
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters) - 1
    unique_lables = list(set(clusters))

    X_scaled[:, 0] = (X_scaled[:, 0]) * 480
    X_scaled[:, 1] = (X_scaled[:, 1]) * 640

    # Generate colors using OpenCV colormap
    colors = [cv2.applyColorMap(np.uint8([[i * 255 // max(1, num_clusters)]]), cv2.COLORMAP_HSV)[0][0] for i in range(num_clusters + 1)]
    colors = [(128, 128, 128)] + [(int(c[0]), int(c[1]), int(c[2])) for c in colors[1:]]
    colors = [(c[2] / 255, c[1] / 255, c[0] / 255) for c in colors]

    time = events.timestamps()
    time = X_scaled[:, 2]
    x = events.coordinates()[:, 0]
    y = events.coordinates()[:, 1]
    y = 480 - y

    TEMP_PLOT = True
    if TEMP_PLOT:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        for i, cluster in enumerate(unique_lables):
            if cluster != -1:
                mask = clusters == cluster
                ax.scatter(x[mask], y[mask], color=colors[i], s=20)

                points = np.column_stack((x[mask], y[mask]))
                centroid_x, centroid_y = np.mean(points, axis=0)
                max_y = np.max(points[:, 1])  # Highest y-point in the cluster

                # plt.text(centroid_x, centroid_y, f"{circularities[i]:.2f}", fontsize=10, color=colors[i], ha='center')
                plt.text(
                    centroid_x,
                    max_y + 5,
                    f"{circularities[i]:.2f}",
                    fontsize=12,
                    color=colors[i],
                    ha="center",
                )

        # Add a legend with a representative circle
        legend_marker = plt.scatter([], [], color="black", s=30, label="Cluster with circularity value")
        plt.legend(handles=[legend_marker])

        ax.set_xlabel(r"$x$ [px]", fontdict={"size": 15}, labelpad=5)
        ax.set_ylabel(r"$y$ [px]", fontdict={"size": 15}, labelpad=5)

        ax.tick_params(axis="x", labelsize=13, width=1.5)  # Increase size and width for X-axis
        ax.tick_params(axis="y", labelsize=13, width=1.5)  # Increase size and width for Y-axis

        ax.legend(fontsize=14)

        plt.title(
            "DBSCAN Clustering of Dynamic Objects Events (x, y)",
            fontsize=16,
            pad=20,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()


def calculate_circularity(points):
    """
    Calculate the circularity of a set of points.
    Circularity = (4 * pi * Area) / Perimeter^2
    """
    if len(points) < 3:
        return 0

    try:
        hull = ConvexHull(points)
        perimeter = hull.area
        area = hull.volume

        circularity = (4 * np.pi * area) / (perimeter**2)
        return circularity, area, perimeter

    except QhullError:
        # Handle degenerate case (e.g., all points collinear or too few points)
        return 0, 0, 0  # Return a default circularity value


def is_blob_elliptical(contour, threshold=0.2):
    """
    Determines if a blob is elliptical based on contour similarity to a fitted ellipse.

    Parameters:
        contour (ndarray): Contour of the blob.
        threshold (float): Allowed relative difference between the contour and the fitted ellipse areas.

    Returns:
        bool: True if the blob is elliptical, False otherwise.
    """
    # Fit an ellipse to the contour
    if len(contour) < 5:  # fitEllipse requires at least 5 points
        return False

    ellipse = cv2.fitEllipse(contour)

    # Calculate the area of the contour and the ellipse
    contour_area = cv2.contourArea(contour)
    ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)  # π * (semi-major * semi-minor)

    # Compare the areas
    relative_difference = abs(contour_area - ellipse_area) / max(contour_area, ellipse_area)
    return relative_difference


def do_dbscan(
    events_batch,
    dbscan_config,
    time_window,
    K2,
    dist_coeffs2,
    crop_w_init,
    crop_h_init,
    steps_time_window,
    visualizer,
    num_inter_frames=5,
):

    eps = dbscan_config["eps"]
    min_samples = dbscan_config["min_samples"]
    scale_time_fact = dbscan_config["scale_time_fact"]
    circularity_threshold = dbscan_config["circularity_threshold"]
    min_area_cluster = dbscan_config["min_area_cluster"]
    min_perimeter_cluster = dbscan_config["min_perimeter_cluster"]
    max_area_cluster = dbscan_config["max_area_cluster"]
    max_perimeter_cluster = dbscan_config["max_perimeter_cluster"]

    is_ball_detected = False
    ball_event_store = None
    avg_radii = []
    point_centers = []
    timestamps_interm = []
    timestamps_interm_all = []

    X = np.stack(
        (
            events_batch.coordinates()[:, 0],
            events_batch.coordinates()[:, 1],
            events_batch.timestamps(),
            events_batch.polarities(),
        ),
        axis=1,
    )

    X_scaled = copy.deepcopy(X[:, :-1]).astype(np.float64)
    X_scaled[:, 0] = X_scaled[:, 0] / 640
    X_scaled[:, 1] = X_scaled[:, 1] / 480
    X_scaled[:, 2] = (X_scaled[:, 2] - np.min(X_scaled[:, 2])) / (np.max(X_scaled[:, 2]) - np.min(X_scaled[:, 2]))
    X_scaled[:, -1] = X_scaled[:, -1] / (time_window * scale_time_fact)

    ball_center = (320, 240)
    ball_center = (((ball_center[0] + crop_w_init) / 640), (((ball_center[1] + crop_h_init)) / 480))

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    unique_labels = set(clusters)
    max_circularity = 0

    circularities = []
    idx_clustersss = []
    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = X[:, :2][clusters == label]

        if cluster_points.shape[0] > 3:
            circularity, area, perimeter = calculate_circularity(cluster_points)
        else:
            continue
        
        circularities.append(circularity)
        idx_clustersss.append(label)
        if circularity > max_circularity and circularity > circularity_threshold and area > min_area_cluster and perimeter > min_perimeter_cluster and area < (max_area_cluster + max(0, (steps_time_window - 24)) * 5) and perimeter < max_perimeter_cluster:
            max_circularity = circularity
            best_label = label
            is_ball_detected = True

    if is_ball_detected:

        idx_cluster = best_label
        X_scaled = X_scaled[clusters == idx_cluster]
        X_ball = X[clusters == idx_cluster]
        clusters = clusters[clusters == idx_cluster]

        ball_event_store = dv.EventStore()
        for ev in X_ball:
            undist_pt = cv2.undistortPoints(np.array([[[int(ev[0]), int(ev[1])]]], dtype=np.float32), K2, dist_coeffs2)

            ball_event_store.push_back(
                ev[2],
                int((undist_pt[0][0][0] * K2[0, 0]) + K2[0, 2]),
                int((undist_pt[0][0][1] * K2[1, 1]) + K2[1, 2]),
                bool(ev[3]),
            )

        tstart = ball_event_store.timestamps()[0]
        tend = ball_event_store.timestamps()[-1]

        timestamps_interm_list = np.linspace(tstart, tend, num=num_inter_frames + 1)
        avg_radii = []
        point_centers = []
        timestamps_interm = []
        timestamps_interm_all = []
        j = 0

        for k in range(len(timestamps_interm_list) - 1):

            ev_intermediate = ball_event_store.sliceTime(int(timestamps_interm_list[k]), int(timestamps_interm_list[k + 1]))
            j += 1

            points_circle_list = []
            for ev_int in ev_intermediate:
                points_circle_list.append([ev_int.x(), ev_int.y()])

            if len(points_circle_list) <= 2:
                break

            points_circle = np.array(points_circle_list)

            mask = np.zeros((480, 640), dtype=np.uint8)
            for coord in points_circle:
                x, y = coord
                if 0 <= x < 640 and 0 <= y < 480:
                    mask[y, x] = 255

            closed_mask = mask
            non_zero_points = cv2.findNonZero(closed_mask)
            points_circle = np.squeeze(non_zero_points)

            if points_circle.size <= 2:
                break

            if points_circle.shape[0] <= 2:
                break

            xc, yc, r, sigma = taubinSVD(points_circle)
            center, radii, orientation = fit_ellipse(points_circle)
            xc_ransac, yc_ransac, radius = ransac_circle_with_density(points_circle, num_iterations=1)

            if (xc_ransac is None) or (yc_ransac is None) or (radius is None):
                break

            if math.isinf(xc) or math.isinf(yc) or math.isinf(center[0]) or math.isinf(center[1]) or math.isinf(xc_ransac) or math.isinf(yc_ransac):
                break

            point_centers.append((xc_ransac, yc_ransac))
            avg_radii.append(radius)

            timestamps_interm.append((((timestamps_interm_list[k] + timestamps_interm_list[k + 1]) / 2) * 1e-6))
            timestamps_interm_all.append((((timestamps_interm_list[k] + timestamps_interm_list[k + 1]) / 2) * 1e-6))

    return (
        point_centers,
        avg_radii,
        timestamps_interm,
        timestamps_interm_all,
        is_ball_detected,
        ball_event_store,
    )
