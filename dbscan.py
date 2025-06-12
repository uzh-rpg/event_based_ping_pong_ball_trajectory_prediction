import cv2
import copy
import math
import numpy as np
import dv_processing as dv
from sklearn.cluster import DBSCAN
from circle_fitting import *
from scipy.spatial import ConvexHull, QhullError


def calculate_circularity(set_points):
    # Calculate the circularity of a set of points using Convex Hull
    if len(set_points) < 3:
        return 0, 0, 0
    
    # Try to compute the convex hull and area (handle degenerate case in 
    # which points are collinear, thrown by QhullError)
    try:
        # Compute the convex hull of the points
        hull = ConvexHull(set_points)
        perimeter = hull.area
        area = hull.volume

        # Calculate circularity: 4 * pi * area / perimeter^2
        circularity = (4 * np.pi * area) / (perimeter**2)
        
        return circularity, area, perimeter
    
    except QhullError:
        return 0, 0, 0  # Return a default circularity value


def do_dbscan(
    events, event_img_resolution,
    dbscan_config, time_window, num_detections, K_events, D_events):

    eps = dbscan_config["eps"]
    min_samples = dbscan_config["min_samples"]
    scale_time_factor = dbscan_config["scale_time_factor"]
    circularity_threshold = dbscan_config["circularity_threshold"]
    min_area_cluster = dbscan_config["min_area_cluster"]
    min_perimeter_cluster = dbscan_config["min_perimeter_cluster"]
    max_area_cluster = dbscan_config["max_area_cluster"]
    max_perimeter_cluster = dbscan_config["max_perimeter_cluster"]

    is_ball_detected = False
    ball_events = None
    ball_center_measurements = []
    ball_radius_measurements = []
    timestamp_measurements = []

    # Stack the event data into a 2D array
    events_data = np.stack(
                    ( events.coordinates()[:, 0],
                    events.coordinates()[:, 1],
                    events.timestamps(),
                    events.polarities() ), axis=1)

    # Normalize the coordinates and timestamps in the [0, 1] range
    events_data_normalized = copy.deepcopy(events_data[:, :-1]).astype(np.float64)
    events_data_normalized[:, 0] = events_data_normalized[:, 0] / event_img_resolution[0]
    events_data_normalized[:, 1] = events_data_normalized[:, 1] / event_img_resolution[1]
    events_data_normalized[:, 2] = (events_data_normalized[:, 2] - np.min(events_data_normalized[:, 2])) / (np.max(events_data_normalized[:, 2]) - np.min(events_data_normalized[:, 2]))
    events_data_normalized[:, -1] = events_data_normalized[:, -1] / (time_window * scale_time_factor)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(events_data_normalized)

    unique_labels = set(clusters)
    circularities = []
    idx_clusters = []
    max_circularity = 0
    
    # Iterate through each unique cluster label and calculate circularity
    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = events_data[:, :2][clusters == label]

        # If the cluster has less than 3 points, skip it
        if cluster_points.shape[0] > 3:
            circularity, area, perimeter = calculate_circularity(cluster_points)
        else:
            continue
        
        circularities.append(circularity)
        idx_clusters.append(label)
        
        # Check if the circularity is greater than the current maximum, in order to find 
        # the cluster with the highest circularity
        if circularity > max_circularity and circularity > circularity_threshold and \
                            area > min_area_cluster and perimeter > min_perimeter_cluster and \
                                area < max_area_cluster and perimeter < max_perimeter_cluster:
                                    
            max_circularity = circularity
            best_label = label
            is_ball_detected = True

    # If a cluster with sufficient circularity was found, extract the points and create an event store
    if is_ball_detected:

        # Filter the points belonging to the best cluster
        idx_cluster = best_label
        events_data_normalized = events_data_normalized[clusters == idx_cluster]
        events_ball = events_data[clusters == idx_cluster]
        clusters = clusters[clusters == idx_cluster]

        # Construct the event store for the detected ball events
        ball_events = dv.EventStore()
        for ev in events_ball:
            undist_pt = cv2.undistortPoints(np.array([[[int(ev[0]), int(ev[1])]]], dtype=np.float32), K_events, D_events)

            ball_events.push_back(
                ev[2],
                int((undist_pt[0][0][0] * K_events[0, 0]) + K_events[0, 2]),
                int((undist_pt[0][0][1] * K_events[1, 1]) + K_events[1, 2]),
                bool(ev[3]),
            )

        tstart = ball_events.timestamps()[0]
        tend = ball_events.timestamps()[-1]
        timestamp_measurements_list = np.linspace(tstart, tend, num=num_detections + 1)

        # Iterate through the timestamps to extract intermediate events
        for k in range(len(timestamp_measurements_list) - 1):
            ev_intermediate = ball_events.sliceTime(
                int(timestamp_measurements_list[k]), int(timestamp_measurements_list[k + 1])
            )

            points_circle_list = []
            for ev_int in ev_intermediate:
                points_circle_list.append([ev_int.x(), ev_int.y()])

            points_circle = np.array(points_circle_list)

            # Create a mask for the points in the circle
            mask = np.zeros((event_img_resolution[1], event_img_resolution[0]), dtype=np.uint8)
            for coord in points_circle:
                # Ensure the coordinates are within the image resolution bounds
                x, y = coord
                if 0 <= x < event_img_resolution[0] and 0 <= y < event_img_resolution[1]:
                    mask[y, x] = 255

            points_circle = np.squeeze(cv2.findNonZero(mask))

            # If there are not enough points to fit a circle, break the loop
            if (points_circle.size <= 2) or (points_circle.shape[0] <= 2):
                break

            # Estimate the circle using RANSAC by fitting a circle to the points and maximizing inliers and density
            xc_ransac, yc_ransac, radius = ransac_circle_estimation(points_circle, num_iterations=1)

            # If the RANSAC estimation fails or returns invalid values, break the loop
            if (xc_ransac is None) or (yc_ransac is None) or (radius is None) or \
                math.isinf(xc_ransac) or math.isinf(yc_ransac) or math.isinf(radius):
                break

            ball_center_measurements.append((xc_ransac, yc_ransac))
            ball_radius_measurements.append(radius)
            timestamp_measurements.append((((timestamp_measurements_list[k] + timestamp_measurements_list[k + 1]) / 2) * 1e-6))

    return ball_center_measurements, ball_radius_measurements, timestamp_measurements, is_ball_detected, ball_events
