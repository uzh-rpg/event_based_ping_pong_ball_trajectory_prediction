import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2

import matplotlib

matplotlib.use("TkAgg")  # or 'Qt5Agg', 'TkAgg', etc.

import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import yaml


import dv_processing as dv

import numpy as np
import copy
from scipy.spatial.transform import Rotation as Rot

from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.stream_id import StreamId, RecordableTypeId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection, get_nearest_eye_gaze

from scipy.ndimage import maximum_filter

import traj_pred.trajectory as traj
import traj_pred.utils as utils
import argparse
import logging

from ekf import *
from motion_compensation import NormalizedMeanTimestampImage
from circle_fitting import *
from dbscan import do_dbscan
from trajectory_prediction import (
    monotonically_constrained_regression,
    predict_trajectory_DCGM,
    compute_trajectory_from_position_untilBOUNCE,
)
from utils.rigid_transformations import *
from utils.visualization import *
from utils.camera import *

logging.basicConfig(level=logging.INFO)


def load_event_data_dvs(reader):
    # Get and print the camera name that data from recorded from
    print(f"Opened an AEDAT4 file which contains data from [{reader.getCameraName()}] camera")

    # Check if event stream is available
    if reader.isEventStreamAvailable():
        resolution = reader.getEventResolution()
        print(f"  * Event stream with resolution [{resolution[0]}x{resolution[1]}]")

    # Initialize an empty store
    events_store = dv.EventStore()

    # Run the loop while camera is still connected
    while reader.isRunning():
        # Read batch of events
        events = reader.getNextEventBatch()
        if events is not None:
            events_store.add(events)

    return events_store


def project_points_DVS(point_3D, K, dist_coeffs):
    point_projected_2D, _ = cv2.projectPoints(point_3D, rvec=np.zeros(3), tvec=np.zeros(3), cameraMatrix=K, distCoeffs=dist_coeffs)

    point_projected_2D = point_projected_2D.squeeze()

    undistorted_point = cv2.undistortPoints(
        np.array([[[int(point_projected_2D[0]), int(point_projected_2D[1])]]], dtype=np.float32),
        K,
        np.zeros((1, 4)),
    )
    X, Y = undistorted_point[0][0]
    point_unprojected_3D = np.array([-X, Y, 1])

    return point_unprojected_3D, point_projected_2D


def transform_point_dvs_aria(point_unprojected, R_quat, t):
    transformation_W_to_A = np.eye(4)
    transformation_W_to_A[:3, :3] = np.linalg.inv(Rot.from_quat(R_quat).as_matrix())  # Rotation part
    transformation_W_to_A[:3, 3] = -t
    transformation_A_to_W = np.linalg.inv(transformation_W_to_A)
    position_B_in_A_homogeneous = transformation_A_to_W @ np.append(point_unprojected, 1)
    position_B_in_A = position_B_in_A_homogeneous[:3]
    point_in_camera_1 = position_B_in_A.reshape((3, 1))

    point_in_camera_1_axis_inv = copy.deepcopy(point_in_camera_1)
    point_in_camera_1_axis_inv[0] = point_in_camera_1[1]
    point_in_camera_1_axis_inv[1] = point_in_camera_1[0]

    return point_in_camera_1_axis_inv


def compensate_rotated_calibration(points_2d, rotation_matrix_eyegaze, t_eyegaze, P1, K2, D2):
    undistorted_point = cv2.undistortPoints(np.array([[[int(points_2d[0]), int(points_2d[1])]]], dtype=np.float32), K2, D2)
    X, Y = undistorted_point[0][0]

    point_unprojected_3D = np.array([X, Y, 1])
    point_unprojected_3D_rotated = np.linalg.inv(rotation_matrix_eyegaze) @ (point_unprojected_3D - t_eyegaze)
    image_point_rotated = P1.project(point_unprojected_3D_rotated)

    return image_point_rotated



def get_filtered_events_store(blurred_mask, input_events, gaze_pt_event_frame, window_gaze_crop, visualizer, accumulator):
    
    blurred_mask = (blurred_mask - np.min(blurred_mask)) / (np.max(blurred_mask) - np.min(blurred_mask))
    blurred_mask *= 255

    frame = visualizer.generateImage(input_events)
    frame_filtered = copy.deepcopy(frame)
    frame_filtered[:, :, 0] = np.where(blurred_mask != 0, frame[:, :, 0], 255)
    frame_filtered[:, :, 1] = np.where(blurred_mask != 0, frame[:, :, 1], 255)
    frame_filtered[:, :, 2] = np.where(blurred_mask != 0, frame[:, :, 2], 255)

    filtered_event_store = dv.EventStore()
    event_pixels = np.where(np.any(frame_filtered != [255, 255, 255], axis=-1))
    filtered_event_coords = np.column_stack((event_pixels[0], event_pixels[1]))
    filtered_event_coords = np.flip(filtered_event_coords, axis=1)

    original_event_coords = input_events.coordinates()
    original_event_timestamps = input_events.timestamps()
    original_event_polarities = input_events.polarities()

    matching_indices = []
    events_filt = []
    for i, point in enumerate(filtered_event_coords):
        matches = np.where(np.all(original_event_coords == point, axis=1))[0]
        assert len(matches) >= 1

        for match_idx in matches:
            matching_indices.extend(matches)

            events_filt.append(
                [
                    int(original_event_timestamps[match_idx]),
                    original_event_coords[match_idx, 0],
                    original_event_coords[match_idx, 1],
                    original_event_polarities[match_idx],
                ]
            )

    events_filt = np.array(events_filt)
    events_filt = events_filt[events_filt[:, 0].argsort()]

    for ev in events_filt:
        filtered_event_store.push_back(ev[0], ev[1], ev[2], bool(ev[3]))

    num_events_obj = len(filtered_event_store.timestamps())
    crop_h_init = max(0, int(gaze_pt_event_frame[1] - window_gaze_crop))
    crop_w_init = max(0, int(gaze_pt_event_frame[0] - window_gaze_crop))

    return filtered_event_store, num_events_obj, crop_w_init, crop_h_init


def crop_image_around_point(img, center, crop_size):
    """Crop an image around a center point with a given half-window size."""
    y, x = int(center[1]), int(center[0])
    h, w = img.shape[:2]
    y1 = max(0, y - crop_size)
    y2 = min(h, y + crop_size)
    x1 = max(0, x - crop_size)
    x2 = min(w, x + crop_size)
    return img[y1:y2, x1:x2]

def crop_events_with_filter(events_batch, gaze_pt, crop_size, img_shape):
    """Apply a dv.EventRegionFilter or similar filter to an event batch and return the cropped events."""
    x, y = int(gaze_pt[0]), int(gaze_pt[1])
    w, h = img_shape[1], img_shape[0]
    x1 = max(0, x - crop_size)
    y1 = max(0, y - crop_size)
    x2 = min(w, x + crop_size)
    y2 = min(h, y + crop_size)
    
    # Calculate width and height of the crop region
    width = x2 - x1
    height = y2 - y1
    
    # Create the event filter with the specified region
    event_filter = dv.EventRegionFilter((x1, y1, width, height))
    
    # Apply the filter to the events batch
    event_filter.accept(events_batch)
    
    return event_filter.generateEvents()


def draw_projected_points_on_image(img, points_3d, cam_calibration, color=(0, 255, 0), radius=5):
    """Draw projected 3D points on an image."""
    for pt in points_3d:
        uv = cam_calibration.project(pt)
        if uv is not None:
            cv2.circle(img, (int(uv[0]), int(uv[1])), radius, color, -1)
        else:
            break
    return img

def compute_gt_points_world(transformed_points, cam_calibration, transform_camera_device, aria_glasses_odometry, idx_aria_odom):
    """Compute ground truth points in world coordinates."""
    gt_points_world = []
    for pt in transformed_points:
        pt_proj = cam_calibration.project(pt)
        pt_rot = copy.deepcopy(pt_proj)
        pt_rot[0] = pt_proj[1]
        pt_rot[1] = cam_calibration.get_image_size()[0] - pt_proj[0]
        point_unproj_3d = cam_calibration.unproject((int(pt_rot[0]), int(pt_rot[1])))
        point_unproj_3d *= pt[-1]
        gt_point_device = np.linalg.inv(transform_camera_device[0:3, 0:3]) @ (
            point_unproj_3d.reshape(3, 1) - transform_camera_device[0:3, 3].reshape(3, 1)
        )
        point_world = aria_glasses_odometry[idx_aria_odom][1] @ gt_point_device.reshape(3, 1)
        gt_points_world.append(point_world)
    return np.array(gt_points_world)[:, :, 0]

def draw_transformed_points_on_img(
    img,
    points_world,
    R_quat,
    t,
    aria_rot,
    translation,
    transform_camera_device,
    cam_calibration,
    rotation_matrix_eyegaze,
    t_eyegaze,
    K2,
    dist_coeffs2,
    color,
    radius=3,
    is_estimated_traj=False,
    is_new_dataset=False,
):
    """Transform world points, project to ARIA image, and draw them."""
    points_DVS, _ = transform_points_from_world_to_DVS_withTransl(
        points_world,
        R_quat,
        t,
        aria_rot,
        translation,
        transform_camera_device,
        cam_calibration,
        rotation_matrix_eyegaze,
        t_eyegaze,
        is_estimated_traj=is_estimated_traj,
        is_new_dataset=is_new_dataset,
    )
    for pt in points_DVS:
        point_unprojected, point_projected_2D = project_points_DVS(pt.reshape(1, 3), K2, dist_coeffs2)
        point_in_camera_1 = transform_point_dvs_aria(point_unprojected, R_quat, t)
        image_point_1 = cam_calibration.project(point_in_camera_1)
        if rotation_matrix_eyegaze is not None:
            image_point_1 = compensate_rotated_calibration(
                point_projected_2D,
                rotation_matrix_eyegaze,
                t_eyegaze,
                cam_calibration,
                K2,
                dist_coeffs2,
            )
        if image_point_1 is not None:
            point_aria_repr = image_point_1.ravel()[:2]
            cv2.circle(
                img,
                (int(point_aria_repr[0]), int(point_aria_repr[1])),
                radius,
                color,
                -1,
            )
    return img

def run_ball_trajectory_prediction(args, config):
    
    visualizer = dv.visualization.EventVisualizer((640, 480))
    visualizer.setBackgroundColor(dv.visualization.colors.white())
    visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
    visualizer.setNegativeColor(dv.visualization.colors.darkGrey())

    # dynamics
    dynamics_config = config["dynamics"]
    
    # ekf bootstrapping
    ekf_config = config["ekf"]

    # pipeline
    pipeline_config = config["pipeline"]
    step_time_interval = pipeline_config["step_time_interval"]
    do_single_batch_eval = pipeline_config["do_single_batch_eval"]
    is_new_dataset = pipeline_config["is_new_dataset"]
    single_detection_time = pipeline_config["single_detection_time"]
    window_gaze_crop = pipeline_config["window_gaze_crop"]

    # calibration
    calibration_config = config["calibration"]
    pos_ARIA_RGB = calibration_config["pos_ARIA_RGB"]
    quat_ARIA_RGB = calibration_config["quat_ARIA_RGB"]
    K2 = np.array(calibration_config["K_events"])
    dist_coeffs2 = np.array(calibration_config["D_events"])
    R_quat = np.array([-0.10873907, -0.04033879, -0.00103722, 0.99325099])
    t = np.array([0.032368, 0.01078887, 0.00610135])

    rotation_matrix_eyegaze = np.array(calibration_config.get("rotation_matrix_eyegaze", None)) if calibration_config.get("rotation_matrix_eyegaze") is not None else None
    t_eyegaze = np.array(calibration_config.get("t_eyegaze", None)) if calibration_config.get("t_eyegaze") is not None else None

    # DGCM model loading
    model_path = config.get("model", {}).get("path_to_model", None)
    model = traj.load_model(model_path)

    # DBSCAN params
    dbscan_config = config["DBSCAN"]
    
    # Motion Compensation
    thresh_0 = config["motion_compensation"]["thresh_0"]
    thresh_1 = config["motion_compensation"]["thresh_1"]

    path_parts = args.path_input_sequence.split("/")
    sequence_id = path_parts[-1] if path_parts[-1] else path_parts[-2]
    sequence_id = int(sequence_id)
    
    # Load the events from the DVS camera
    events_filename = "events.aedat4"
    reader = dv.io.MonoCameraRecording(os.path.join(args.path_input_sequence, events_filename))
    events = load_event_data_dvs(reader)
    timestamps_events = events.timestamps()


    # Load the Aria Glasses VRS file
    name_vrs = "aria_recording.vrs"
    mps_sample_path = os.path.dirname(args.path_input_sequence)
    vrsfile = os.path.join(mps_sample_path, name_vrs)
    aria_glasses_odometry = []
    aria_glasses_odometry_timestamps = []

    # Loading GT trajectories
    ground_truth = np.load(os.path.join(args.path_input_sequence, "trajectories.npz"))
    traj_timestamps_i = ground_truth["traj_timestamps"] * 1e9
    traj_timestamps_i_full = copy.deepcopy(traj_timestamps_i)
    traj_timestamps_i_dt = traj_timestamps_i - traj_timestamps_i[0]
    traj_3D_points_i = ground_truth["traj_points_W"]
    traj_3D_points_i_full = copy.deepcopy(traj_3D_points_i)

    timestamp_body_W_i = ground_truth["aria_pose_timestamps"] * 1e9
    pos_W_ARIA_i = ground_truth["aria_positions_W"]
    quat_W_ARIA_i = ground_truth["aria_orientations_W"]
    imu_data_i = ground_truth["imu_data_DVS_i"]
    is_bouncing = ground_truth["is_bouncing"]

    # Loading Aria poses (whole sequence)
    aria_pose_timestamps = copy.deepcopy(timestamp_body_W_i)
    aria_positions = copy.deepcopy(pos_W_ARIA_i)
    aria_orientations = copy.deepcopy(quat_W_ARIA_i)

    imu_data_new = copy.deepcopy(imu_data_i)
    gyro_data = imu_data_new[:, -3:]

    timestamps_aria_rgb = np.loadtxt(os.path.join(args.path_input_sequence, "timestamps_ns.txt"), dtype=int)
    timestamps_aria_rgb_dt = timestamps_aria_rgb - timestamps_aria_rgb[0]

    pos_ARIA_RGB = np.array([-0.0187454, -0.0627825, -0.0734011])
    quat_ARIA_RGB = np.array([0.715736, -0.0258874, 0.697549, 0.0218361])


    # Get the device calibration and camera calibration from Aria VRS fil
    provider = data_provider.create_vrs_data_provider(vrsfile)
    provider.get_device_calibration().get_transform_device_sensor("camera-rgb")

    transform_device_camera = provider.get_device_calibration().get_transform_device_sensor("camera-rgb").to_matrix()
    transform_camera_device = np.linalg.inv(transform_device_camera)

    # Load eyegaze, trajectory and global points
    generalized_eye_gaze_path = os.path.join(mps_sample_path, "eye_gaze", "general_eye_gaze.csv")
    closed_loop_trajectory = os.path.join(mps_sample_path, "slam", "closed_loop_trajectory.csv")

    generalized_eye_gazes = mps.read_eyegaze(generalized_eye_gaze_path)
    mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory)

    for pose in mps_trajectory:
        aria_glasses_odometry_timestamps.append(pose.tracking_timestamp.total_seconds() * 1e9)
        aria_glasses_odometry.append(
            [
                pose.transform_world_device.translation(),
                pose.transform_world_device.rotation().to_matrix(),
            ]
        )

    ###################################################################
    ###################################################################

    # Default options activates all streams
    options = provider.get_default_deliver_queued_options()
    options.deactivate_stream_all()

    time_domain = TimeDomain.DEVICE_TIME  # query data based on host time
    option = TimeQueryOptions.CLOSEST  # get data whose time [in TimeDomain] is CLOSEST to query time

    rgb_stream_ids = options.get_stream_ids(RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS)
    left_imu_stream_id = provider.get_stream_id_from_label("imu-left")

    options.activate_stream(left_imu_stream_id)
    options.activate_stream(rgb_stream_ids[0])

    iterator = provider.deliver_queued_sensor_data(options)
    imu_left_device_timestamps = []
    rgb_camera_device_timestamps = []

    for sensor_data in iterator:
        label = provider.get_label_from_stream_id(sensor_data.stream_id())
        device_timestamp = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)

        if label == "camera-rgb":
            rgb_camera_device_timestamps.append(device_timestamp)

        if label == "imu-left":
            imu_left_device_timestamps.append(device_timestamp)

    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)

    device_calibration = provider.get_device_calibration()
    cam_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    

    points_ = []
    for y in range(480):
        for x in range(640):
            points_.append((x, y))

    points_ = np.array(points_, dtype=np.float32).reshape(-1, 1, 2)
    rectified_points_ = cv2.undistortPoints(points_, K2[:3, :3], dist_coeffs2)
    normalized_image = NormalizedMeanTimestampImage(480, 640, K2[:3, :3], rectified_points_)

    ###############################################################################################
    ################################################################################################
    ################################################################################################

    trajectory_estimates = []
    ball_radii_estimates = []
    timestamps_trajetory_interm = []

    step = step_time_interval
    num_int_frame_dbscan = int(step / single_detection_time)
    
    cum_step = 0
    cum_num_int_frame_dbscan = 0

    accumulator = dv.Accumulator(reader.getEventResolution())

    point_centers_kf = []
    avg_radii_kf = []
    timestamps_kf = []
    timestamps_kf_all = []

    cum_dt = 0
    cum_iter_count = 0

    trajectory_positions = None
    img_aria_temp = None

    t_start_pipeline = 0.01
    lenght_trajectory = 0.591666
    t_imu = copy.deepcopy(imu_data_new[:, 0])
    t_imu -= t_imu[0]
    t_imu *= 1e-6
    idx_start_imu = np.where(t_imu > t_start_pipeline)[0][0]
    imu_data_new = imu_data_new[idx_start_imu:, :]

    cum_iter_count = len(imu_data_new[:idx_start_imu, 0])
    cum_dt_count = t_start_pipeline

    for i in range(0, len(imu_data_new) - step - 1, step):

        cum_iter_count += step

        t0 = int(imu_data_new[i, 0])
        t1 = int(imu_data_new[i + step, 0])
        dt = (t1 - t0) * 1e-6
        cum_dt_count += dt
        events_batch = events.sliceTime(t0, t1)
        
        # Start processing only if enough time has passed
        if cum_dt_count < t_start_pipeline:
            continue

        events_batch = events.sliceTime(t0, t1)

        # Get event image and accumulate events
        accumulator.accept(events_batch)
        frame = accumulator.generateFrame()
        img_event = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)

        # Get eye gaze reprojection point
        index = np.abs(timestamps_aria_rgb_dt - ((t0 - timestamps_events[0]) * 1e3)).argmin()

        traj_timestamps_i_dt_sec = [t * 1e-9 for t in traj_timestamps_i_dt]
        index_traj = np.abs(traj_timestamps_i_dt_sec - ((t0 - timestamps_events[0]) * 1e-6)).argmin()

        capture_timestamp_ns = provider.get_image_data_by_time_ns(rgb_stream_id, int(timestamps_aria_rgb[index]), time_domain, option)[1].capture_timestamp_ns
        generalized_eye_gaze = get_nearest_eye_gaze(generalized_eye_gazes, capture_timestamp_ns)
        
        depth_m = generalized_eye_gaze.depth or 1.0
        generalized_gaze_center_in_pixels = get_gaze_vector_reprojection(
            generalized_eye_gaze,
            rgb_stream_label,
            device_calibration,
            cam_calibration,
            depth_m,
        )
        
        # Reproject gaze point in event frame
        gaze_pt_event_frame = reproject_pt_in_event_frame(cam_calibration, img_event, generalized_gaze_center_in_pixels, R_quat, t, K2, dist_coeffs2)

        if rotation_matrix_eyegaze is not None:
            eye_gaze_aria_unproj = cam_calibration.unproject((int(generalized_gaze_center_in_pixels[0]), int(generalized_gaze_center_in_pixels[1])))
            point_3d_cam2 = rotation_matrix_eyegaze @ eye_gaze_aria_unproj + t_eyegaze
            gaze_pt_event_frame, _ = cv2.projectPoints(point_3d_cam2.reshape((1, 3)), np.zeros((3, 1)), np.zeros((3, 1)), K2, dist_coeffs2)
            gaze_pt_event_frame = gaze_pt_event_frame[0, 0]

        idx_aria_odom = min(
            range(len(aria_glasses_odometry_timestamps)),
            key=lambda k: abs(aria_glasses_odometry_timestamps[k] - timestamps_aria_rgb[index]),
        )

        if len(events_batch.timestamps()) > 0:

            # Crop the events around the gaze point
            cropped_events_batch = crop_events_with_filter(events_batch, gaze_pt_event_frame, window_gaze_crop, img_event.shape)
            
            cum_dt += dt

            if len(cropped_events_batch.timestamps()) <= 0:
                continue

            # Apply motion compensation
            w_body = np.sum(gyro_data[i : (i + step)], axis=0) / step
            _, num_events = normalized_image.draw_normalized_mean_timestamp_image(cropped_events_batch, -w_body)

            if (np.sum(num_events) == 0):
                continue

            num_events_mc = np.sum(num_events)
            
            # Normalize the image and apply mean filter
            normalized_image.mean_filter_image()
            result = normalized_image.normalized_mean_timestamp_image
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
            result *= 255
            cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Filter out events based on the motion compensation threshold
            warp_norm = np.sqrt(w_body[0] ** 2 + w_body[1] ** 2 + w_body[2] ** 2)
            threshold = min((thresh_0 * warp_norm + thresh_1) * 255, 255)
            result[result < threshold] = 0

            # Apply Gaussian blur to the result and get events 
            blurred_mask = gaussian_filter(result, sigma=1.2)
            filtered_event_store, num_events_obj, crop_w_init, crop_h_init = get_filtered_events_store(blurred_mask, cropped_events_batch, gaze_pt_event_frame, window_gaze_crop, visualizer, accumulator)

            # Run DBSCAN 
            if trajectory_positions is None:
                (
                    point_centers,
                    avg_radii,
                    timestamps_interm,
                    timestamps_interm_all,
                    is_ball_detected,
                    ball_events,
                ) = do_dbscan(
                    filtered_event_store,
                    dbscan_config,
                    step,
                    K2,
                    dist_coeffs2,
                    crop_w_init,
                    crop_h_init,
                    step_time_interval,
                    visualizer,
                    num_inter_frames=num_int_frame_dbscan,
                )

            if (len(point_centers) == 0 or (not is_ball_detected)):
                continue

            if ball_events is not None:
                num_events_b = len(ball_events.timestamps())

            if is_ball_detected:
                if trajectory_positions is None:

                    point_centers_kf.extend(point_centers)
                    avg_radii_kf.extend(avg_radii)
                    timestamps_kf.extend(timestamps_interm)
                    timestamps_kf_all.extend(timestamps_interm_all)

                    frame = visualizer.generateImage(ball_events)

                for d in range(len(timestamps_interm)):
                    ball_radii_estimates.append(avg_radii[d])
                    trajectory_estimates.append([point_centers[d][0], point_centers[d][1]])
                    timestamps_trajetory_interm.append(timestamps_interm[d])

                # If we have enough trajectory estimates, we can proceed with the trajectory prediction
                if len(trajectory_estimates) >= 2:

                    t_horizon = (lenght_trajectory - cum_dt_count) * 1e9
                    idx_aria_odom = min(
                        range(len(aria_glasses_odometry_timestamps)),
                        key=lambda k: abs(aria_glasses_odometry_timestamps[k] - traj_timestamps_i_full[index_traj]),
                    )

                    transformed_points_over_time = transform_traj_wrt_pose(
                        traj_3D_points_i_full,
                        aria_positions[np.abs(aria_pose_timestamps - traj_timestamps_i_full[index_traj]).argmin()],
                        aria_orientations[np.abs(aria_pose_timestamps - traj_timestamps_i_full[index_traj]).argmin()],
                        pos_ARIA_RGB,
                        quat_ARIA_RGB,
                    )

                    traj_3D_points_i = traj_3D_points_i_full[traj_timestamps_i_full >= traj_timestamps_i_full[index_traj]]
                    transformed_points_over_time = transformed_points_over_time[traj_timestamps_i_full >= traj_timestamps_i_full[index_traj]]
                    traj_timestamps_i = traj_timestamps_i_full[traj_timestamps_i_full >= traj_timestamps_i_full[index_traj]]

                    img_rgb = provider.get_image_data_by_time_ns(rgb_stream_id, int(timestamps_aria_rgb[index]), time_domain, option)[0].to_numpy_array()
                    img_rgb = cv2.cvtColor(cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
                    img_rgb = draw_projected_points_on_image(img_rgb, transformed_points_over_time, cam_calibration, color=(0, 255, 0), radius=5)

                    # If any point could not be projected, skip
                    if any(cam_calibration.project(pt) is None for pt in transformed_points_over_time):
                        continue

                    # Compute ground truth points in world coordinates
                    gt_points_world = compute_gt_points_world(
                        transformed_points_over_time, cam_calibration, transform_camera_device, aria_glasses_odometry, idx_aria_odom
                    )

                    dt_gt = np.mean(np.diff(traj_timestamps_i * 1e-9))

                    if do_single_batch_eval:
                        cum_step = step
                        cum_num_int_frame_dbscan = num_int_frame_dbscan
                    else:
                        cum_step += step
                        cum_num_int_frame_dbscan += num_int_frame_dbscan

                    if trajectory_positions is None:
                        initial_p, initial_v, p_estimates, v_estimates = monotonically_constrained_regression(
                            trajectory_estimates,
                            ball_radii_estimates,
                            timestamps_trajetory_interm,
                            aria_glasses_odometry[idx_aria_odom][1],
                            transform_camera_device,
                            R_quat,
                            t,
                            rotation_matrix_eyegaze,
                            t_eyegaze,
                            cum_step,
                            cum_num_int_frame_dbscan,
                            config,
                            is_new_dataset=is_new_dataset,
                        )


                        pred_mean, _ = predict_trajectory_DCGM(
                            model,
                            p_estimates,
                            t_horizon,
                            cum_dt,
                            np.array(timestamps_trajetory_interm),
                        )

                        # Run ODE on the estimated measurements that we got from the ball detection module
                        t_viz_after_bounce = 0.12
                        trajectory_positions, trajectory_timestamps = compute_trajectory_from_position_untilBOUNCE(
                                    initial_p, initial_v, dynamics_config, dt_gt, t_viz_after_bounce)

                        # Bootstrap EKF initial state and compute trajectory
                        measurements = np.hstack([p_estimates[:-1], v_estimates])

                        position_states_EKF, velocity_states_EKF = bootstrap_initial_state_from_EKF(measurements, dt_gt, dynamics_config, ekf_config)

                        init_p_EKF = position_states_EKF[-1]
                        init_v_EKF = np.mean(velocity_states_EKF, axis=0)

                        trajectory_positions_EKF, _ = compute_trajectory_from_position_untilBOUNCE(
                            init_p_EKF, init_v_EKF, dynamics_config, dt_gt, t_viz_after_bounce)

                        trajectory_timestamps += timestamps_trajetory_interm[0]

                        # Extend ground truth trajectory if the current trajectory is not long enough that it bounces on the table
                        dt_gt = np.mean(np.diff(traj_timestamps_i * 1e-9))
                        vel_world = np.diff(gt_points_world, axis=0) / dt_gt
                        initial_v_inf_gt = np.mean(vel_world[-40:], axis=0)
                        initial_p_inf_gt = gt_points_world[-1]

                        position_inf_gt, _ = compute_trajectory_from_position_untilBOUNCE(
                            initial_p_inf_gt, initial_v_inf_gt, dynamics_config, dt_gt, 0.01, True)

                        # Predict trajectory for DCGM model
                        pred_mean = savitzky_golay_filtering(pred_mean)

                        dt_gt = np.mean(np.diff(traj_timestamps_i * 1e-9))
                        vel_world = np.diff(pred_mean, axis=0) / dt_gt
                        initial_v_inf_gt = np.mean(vel_world[-40:], axis=0)
                        initial_p_inf_gt = pred_mean[-1]

                        position_tvae_inf_gt, _ = compute_trajectory_from_position_untilBOUNCE(
                            initial_p_inf_gt, initial_v_inf_gt, dynamics_config, dt_gt, t_viz_after_bounce)

                        # Predict trajectory with ground truth measurements to set an upper bound on the performance
                        t_sample = copy.deepcopy(traj_timestamps_i)
                        t_sample -= t_sample[0]
                        t_sample *= 1e-9

                        dt_gt = np.mean(np.diff(t_sample))
                        index = (np.where(np.array(t_sample) > (step * dt_gt))[0])[0]
                        gt_points_world_UPPER_BOUND = gt_points_world[:index]

                        vel_world = np.diff(gt_points_world_UPPER_BOUND, axis=0) / dt_gt
                        initial_v_gt_upper_bound = np.mean(vel_world, axis=0)
                        initial_p_gt_upper_bound = gt_points_world_UPPER_BOUND[-1]

                        position_gt_upper_bound, _ = compute_trajectory_from_position_untilBOUNCE(
                            initial_p_gt_upper_bound, initial_v_gt_upper_bound, dynamics_config, dt_gt, 0.12)
                        

                        ###########################  PERFORMANCE EVALUATION  #############################

                        gt_points_world_downsampled = downsample_3d_array(gt_points_world, len(trajectory_positions))
                        
                        trajectory_timestamps -= trajectory_timestamps[0]
                        dt_gt = np.mean(np.diff(trajectory_timestamps))

                        # Prepare all intersection arrays and names
                        intersection_configs = [
                            ("diff_eq", trajectory_positions),
                            ("gt", np.vstack((gt_points_world_downsampled, position_inf_gt))),
                            ("upper_bound", np.vstack((gt_points_world_UPPER_BOUND, position_gt_upper_bound))),
                            ("tvae", np.vstack((pred_mean, position_tvae_inf_gt))),
                            ("ekf", trajectory_positions_EKF)
                        ]

                        intersections = {}
                        for name, arr in intersection_configs:
                            min_z_index = np.argmin(arr[:, 2])
                            intersections[name] = arr[min_z_index]

                        # Compute errors in a loop with descriptive names
                        error_pairs = [
                            ("diff_eq", "gt"),
                            ("upper_bound", "gt"),
                            ("tvae", "gt"),
                            ("ekf", "gt")
                        ]
                        error_vars = [
                            ("error_diff_eq_vs_gt", "coord_diff_eq_vs_gt"),
                            ("error_upper_bound_vs_gt", "coord_upper_bound_vs_gt"),
                            ("error_tvae_vs_gt", "coord_tvae_vs_gt"),
                            ("error_ekf_vs_gt", "coord_ekf_vs_gt")
                        ]

                        error_dict = {}
                        coord_dict = {}
                        for (est, gt), (err_name, coord_name) in zip(error_pairs, error_vars):
                            diff = intersections[est][:2] - intersections[gt][:2]
                            error_dict[err_name] = np.linalg.norm(diff)
                            coord_dict[coord_name] = diff           

                        # Print error metrics in a loop for compactness
                        print("SEQUENCE", args.path_input_sequence)
                        error_labels = [
                            "ERROR DIFF EQ VS GT",
                            "ERROR UPPER BOUND VS GT",
                            "ERROR TVAE VS GT",
                            "ERROR EKF VS GT"
                        ]

                        
                        error_values = [
                            error_dict["error_diff_eq_vs_gt"],
                            error_dict["error_upper_bound_vs_gt"],
                            error_dict["error_tvae_vs_gt"],
                            error_dict["error_ekf_vs_gt"]
                        ]
                        
                        for label, value in zip(error_labels, error_values):
                            print(f" {label}: {value}")

                        # Save results
                        np.savez(
                            os.path.join(args.path_input_sequence, "evalutation.npz"),
                            points_GT=gt_points_world,
                            timestamps_GT=traj_timestamps_i,
                            points_PRED=trajectory_positions,
                            timestamps_PRED=trajectory_timestamps,
                            aria_odometry_R=aria_glasses_odometry[idx_aria_odom][1],
                            aria_position_i=aria_positions[np.abs(aria_pose_timestamps - traj_timestamps_i_full[index_traj]).argmin()],
                            aria_orientation_i=aria_orientations[np.abs(aria_pose_timestamps - traj_timestamps_i_full[index_traj]).argmin()],
                        )

                        # Write RMSE row to file
                        def format_performance_evaluation(
                            sequence_id, num_events_mc, num_events_obj, num_events_b,
                            error_diff_eq_vs_gt, coord_diff_eq_vs_gt,
                            error_tvae_vs_gt, coord_tvae_vs_gt,
                            error_upper_bound_vs_gt,
                            error_ekf_vs_gt, coord_ekf_vs_gt,
                            last_timestamp
                        ):
                            return (
                                f"{int(sequence_id)},{int(num_events_mc)},{int(num_events_obj)},{int(num_events_b)},"
                                f"{error_diff_eq_vs_gt},{coord_diff_eq_vs_gt[0]},{coord_diff_eq_vs_gt[1]},"
                                f"{error_tvae_vs_gt},{coord_tvae_vs_gt[0]},{coord_tvae_vs_gt[1]},"
                                f"{error_upper_bound_vs_gt},"
                                f"{error_ekf_vs_gt},{coord_ekf_vs_gt[0]},{coord_ekf_vs_gt[1]},"
                                f"{last_timestamp}\n"
                            )


                        with open(config["io"]["output_path_txt"], "a") as file_rmse:
                            file_rmse.write(format_performance_evaluation(
                                sequence_id, num_events_mc, num_events_obj, num_events_b,
                                error_dict["error_diff_eq_vs_gt"], coord_dict["coord_diff_eq_vs_gt"],
                                error_dict["error_tvae_vs_gt"], coord_dict["coord_tvae_vs_gt"],
                                error_dict["error_upper_bound_vs_gt"],
                                error_dict["error_ekf_vs_gt"], coord_dict["coord_ekf_vs_gt"],
                                timestamps_trajetory_interm[-1]
                            ))
                            
                        
                        ###########################  VISUALIZATION  #############################
                        
                        translation = aria_glasses_odometry[idx_aria_odom][0] - aria_glasses_odometry[idx_aria_odom - 1][0]
                        index = np.abs(timestamps_aria_rgb_dt - ((t0 - timestamps_events[0]) * 1e3)).argmin()
                        img_aria_temp = provider.get_image_data_by_time_ns(rgb_stream_id, int(timestamps_aria_rgb[index]), time_domain, option)[0].to_numpy_array()

                        # Define a boundary circle around the intersection point
                        radius_circle = 0.25
                        theta = np.linspace(0, 2 * np.pi, 150)
                        circle_points = np.array([
                            intersections["gt"][0] + radius_circle * np.cos(theta),
                            intersections["gt"][1] + radius_circle * np.sin(theta),
                            dynamics_config["height_table"] * np.ones_like(theta) ])

                        # List of (points, color) to draw
                        viz_configs = [
                            (circle_points.T, (100, 100, 0)),
                            (trajectory_positions, (255, 0, 0)),
                            (trajectory_positions_EKF, (0, 0, 0)),
                            (np.vstack((gt_points_world_downsampled, position_inf_gt)), (0, 255, 0)),
                        ]
                        for points, color in viz_configs:
                            img_aria_temp = draw_transformed_points_on_img(
                                img_aria_temp,
                                points,
                                R_quat,
                                t,
                                aria_glasses_odometry[idx_aria_odom][1],
                                translation,
                                transform_camera_device,
                                cam_calibration,
                                rotation_matrix_eyegaze,
                                t_eyegaze,
                                K2,
                                dist_coeffs2,
                                color=color,
                                radius=3,
                                is_estimated_traj=False,
                                is_new_dataset=is_new_dataset,
                            )

                        # Legend position and display
                        rotated_img_aria = np.rot90(cv2.cvtColor(cv2.resize(img_aria_temp, (704, 704)), cv2.COLOR_BGR2RGB), -1,)
                        cv2.imshow("Output trajectories (on ARIA frame [propagated])", rotated_img_aria)
                        cv2.waitKey(0)
                                         
                        # Break if only single batch evaluation is required
                        if do_single_batch_eval:
                            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("path_input_sequence", help="Path of the input sequence to test")

    args = parser.parse_args()

    with open(os.path.join(args.path_input_sequence, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    run_ball_trajectory_prediction(args, config)
