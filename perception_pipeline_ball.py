import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import copy
import yaml
import cv2
import dv_processing as dv
from scipy.spatial.transform import Rotation as Rot

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection, get_nearest_eye_gaze

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
    run_differential_equation_trajectory_prediction,
)
from utils.rigid_transformations import *
from utils.visualization import *
from utils.camera import *
from utils.events_utils import *

import matplotlib
matplotlib.use("TkAgg")

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


def run_ball_trajectory_prediction(args, config):  
      
    """LOADING CONFIGURATION PARAMETERS"""
    # Pipeline config parameters
    pipeline_config = config["pipeline"]
    window_gaze_crop = pipeline_config["window_gaze_crop"]
    
    # Dynamics config parameters
    dynamics_config = config["dynamics"]
    
    # EKF bootstrapping config parameters
    ekf_config = config["EKF"]

    # Calibration config parameters
    calibration_config = config["calibration"]
    t_ARIA_RGB = calibration_config["t_ARIA_RGB"]
    R_ARIA_RGB = calibration_config["R_ARIA_RGB"]
    t_RGB_DVS = np.array(calibration_config["t_RGB_DVS"])
    R_RGB_DVS = np.array(calibration_config["R_RGB_DVS"])
    K_events = np.array(calibration_config["K_events"])
    D_events = np.array(calibration_config["D_events"])
    has_rotated_calibration = calibration_config["has_rotated_calibration"]

    R_calib_comp = np.array(calibration_config.get("R_calib_comp", None)) if calibration_config.get("R_calib_comp") is not None else None
    t_calib_comp = np.array(calibration_config.get("t_calib_comp", None)) if calibration_config.get("t_calib_comp") is not None else None

    # DGCM model loading config
    DCGM_model_path = config.get("io", {}).get("path_to_DCGM_model", None)
    model = traj.load_model(DCGM_model_path)

    # DBSCAN config parameters
    dbscan_config = config["DBSCAN"]
    
    # Motion Compensation config parameters
    thresh_0 = config["motion_compensation"]["thresh_0"]
    thresh_1 = config["motion_compensation"]["thresh_1"]

    # Load the events from the DVS camera
    events_filename = "events.aedat4"
    reader = dv.io.MonoCameraRecording(os.path.join(args.path_input_sequence, events_filename))
    events = load_event_data_dvs(reader)
    timestamps_events = events.timestamps()
    
    # Initialize the event visualizer
    visualizer = dv.visualization.EventVisualizer((reader.getEventResolution()[0], reader.getEventResolution()[1]))
    visualizer.setBackgroundColor(dv.visualization.colors.white())
    visualizer.setPositiveColor(dv.visualization.colors.iniBlue())
    visualizer.setNegativeColor(dv.visualization.colors.darkGrey())
    
    # Initialize the accumulator for the events
    accumulator = dv.Accumulator(reader.getEventResolution())

    # Load the Aria Glasses VRS file
    name_vrs = "aria_recording.vrs"
    mps_sample_path = os.path.dirname(args.path_input_sequence)
    vrsfile = os.path.join(mps_sample_path, name_vrs)
    aria_glasses_odometry = []
    aria_glasses_odometry_timestamps = []

    # Loading GT trajectories
    ground_truth = np.load(os.path.join(args.path_input_sequence, "trajectories.npz"))
    trajectory_3D_points = ground_truth["traj_points_W"]
    trajectory_timestamps = ground_truth["traj_timestamps"] * 1e9
    dt = np.mean(np.diff(trajectory_timestamps * 1e-9))

    imu_data = ground_truth["imu_data_DVS_i"]
    gyro_data = imu_data[:, -3:]
    imu_timestamps = imu_data[:, 0]
    imu_timestamps_s = (imu_data[:, 0] - imu_data[:, 0][0]) * 1e-6
    dt_IMU_ms = np.mean(np.diff(imu_timestamps_s)) * 1e3
    is_bouncing = ground_truth["is_bouncing"]

    # Loading Aria poses (whole sequence)
    aria_pose_timestamps = ground_truth["aria_pose_timestamps"] * 1e9
    aria_positions = ground_truth["aria_positions_W"]
    aria_orientations = ground_truth["aria_orientations_W"]

    timestamps_aria_rgb = np.loadtxt(os.path.join(args.path_input_sequence, "timestamps_ns.txt"), dtype=int)
    timestamps_aria_rgb_dt = timestamps_aria_rgb - timestamps_aria_rgb[0]

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
        aria_glasses_odometry.append([pose.transform_world_device.translation(),
                                      pose.transform_world_device.rotation().to_matrix()])

    # Get the RGB and IMU streams from the Aria provider
    time_domain = TimeDomain.DEVICE_TIME
    option = TimeQueryOptions.CLOSEST
    cam_calibration, device_calibration, rgb_stream_id, rgb_stream_label = get_aria_rgb_and_imu_streams(provider)
    
    # Construct the normalized mean timestamp image class
    image_points = []
    for y in range(reader.getEventResolution()[1]):
        for x in range(reader.getEventResolution()[0]):
            image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    rectified_image_points = cv2.undistortPoints(image_points, K_events[:3, :3], D_events)
    normalized_mean_timestamp_image_class = NormalizedMeanTimestampImage(reader.getEventResolution()[1], reader.getEventResolution()[0], K_events[:3, :3], rectified_image_points)

    # Initialize variables for ball detection and trajectory estimation
    total_ball_center_est = []
    total_ball_radius_est = []
    total_measurement_timestamps = []

    num_IMU_samples = int(pipeline_config["batch_accumulation_time_ms"] / dt_IMU_ms)
    num_detections = int(num_IMU_samples / pipeline_config["detection_time_ms"])
    event_counts = {}

    trajectory_positions_ODE = None
    img_aria = None

    # We need to start processing after a certain time to ensure the opponent has hit the ball with the racket
    t_since_start = config["pipeline"]["start_detection_after_s"]
    idx_start_imu = np.where(imu_timestamps_s > config["pipeline"]["start_detection_after_s"])[0][0]
    imu_data = imu_data[idx_start_imu:, :]
    imu_timestamps = imu_timestamps[idx_start_imu:]

    for i in range(0, len(imu_data) - num_IMU_samples, num_IMU_samples):
        # Get the IMU data and event for the current time interval
        t0 = int(imu_timestamps[i])
        t1 = int(imu_timestamps[i + num_IMU_samples])
        
        t_since_start += ((t1 - t0) * 1e-6)
        events_batch = events.sliceTime(t0, t1)

        # Start processing only if enough time has passed
        if t_since_start < config["pipeline"]["start_detection_after_s"]:
            continue

        # We select the events in the current time interval
        events_batch = events.sliceTime(t0, t1)

        # Get event image and accumulate events
        accumulator.accept(events_batch)
        img_event = cv2.cvtColor(accumulator.generateFrame().image, cv2.COLOR_GRAY2BGR)

        # Get eye gaze reprojection point
        index = np.abs(timestamps_aria_rgb_dt - ((t0 - timestamps_events[0]) * 1e3)).argmin()
        trajectory_timestamps_dt = [t * 1e-9 for t in (trajectory_timestamps - trajectory_timestamps[0])]
        index_traj = np.abs(trajectory_timestamps_dt - ((t0 - timestamps_events[0]) * 1e-6)).argmin()

        capture_timestamp_ns = provider.get_image_data_by_time_ns(rgb_stream_id, int(timestamps_aria_rgb[index]), time_domain, option)[1].capture_timestamp_ns
        generalized_eye_gaze = get_nearest_eye_gaze(generalized_eye_gazes, capture_timestamp_ns)
        
        depth_m = generalized_eye_gaze.depth or 1.0
        generalized_gaze_center_in_pixels = get_gaze_vector_reprojection(
                                                                generalized_eye_gaze,
                                                                rgb_stream_label,
                                                                device_calibration,
                                                                
                                                                cam_calibration,
                                                                depth_m)
            
        # Reproject gaze point in event frame
        gaze_pt_event_frame = reproject_ARIA_RGB_to_DVS(cam_calibration, generalized_gaze_center_in_pixels, R_RGB_DVS, t_RGB_DVS, K_events, D_events)

        if R_calib_comp is not None:
            eye_gaze_aria_unproj = cam_calibration.unproject((int(generalized_gaze_center_in_pixels[0]), int(generalized_gaze_center_in_pixels[1])))
            point_3d_cam2 = R_calib_comp @ eye_gaze_aria_unproj + t_calib_comp
            gaze_pt_event_frame, _ = cv2.projectPoints(point_3d_cam2.reshape((1, 3)), np.zeros((3, 1)), np.zeros((3, 1)), K_events, D_events)
            gaze_pt_event_frame = gaze_pt_event_frame[0, 0]

        idx_aria_odom = min(
            range(len(aria_glasses_odometry_timestamps)),
            key=lambda k: abs(aria_glasses_odometry_timestamps[k] - timestamps_aria_rgb[index]),
        )

        if len(events_batch.timestamps()) > 0:
            # Crop the events around the gaze point
            cropped_events_batch = crop_events_with_filter(events_batch, gaze_pt_event_frame, window_gaze_crop, img_event.shape)

            if len(cropped_events_batch.timestamps()) <= 0:
                continue

            # Apply motion compensation
            w_body = np.sum(gyro_data[i : (i + num_IMU_samples)], axis=0) / num_IMU_samples
            _, num_events = normalized_mean_timestamp_image_class.draw_normalized_mean_timestamp_image(cropped_events_batch, -w_body)

            if (np.sum(num_events) == 0):
                continue

            event_counts["num_events_motion_comp"] = np.sum(num_events)

            # Get the normalized mean timestamp image
            normalized_mean_timestamp_image_class.mean_filter_image()
            normalized_mean_timestamp_image = normalized_mean_timestamp_image_class.normalized_mean_timestamp_image

            # Filter out events based on the motion compensation threshold
            warp_norm = np.sqrt(w_body[0] ** 2 + w_body[1] ** 2 + w_body[2] ** 2)
            threshold = thresh_0 * warp_norm + thresh_1
            normalized_mean_timestamp_image[np.abs(normalized_mean_timestamp_image) < threshold] = 0

            # Apply Gaussian blur to the filtered normalized mean timestamp image not to throw away potential events belonging to the ball
            blurred_mask = gaussian_filter(normalized_mean_timestamp_image, sigma=1.2)
            filtered_event_store, num_events_dyn_obj = get_filtered_events_store(blurred_mask, cropped_events_batch, visualizer)
            event_counts["num_events_dyn_obj"] = num_events_dyn_obj

            # Run DBSCAN 
            if trajectory_positions_ODE is None:
                (
                    ball_center_est,
                    ball_radius_est,
                    measurement_timestamps,
                    is_ball_detected,
                    ball_events
                ) = do_dbscan(
                        filtered_event_store,
                        reader.getEventResolution(),
                        dbscan_config,
                        pipeline_config["batch_accumulation_time_ms"],
                        num_detections,
                        K_events,
                        D_events)

            if (len(ball_center_est) == 0 or (not is_ball_detected)):
                continue

            if ball_events is not None:
                event_counts["num_ball_events"] = len(ball_events.timestamps())

            if is_ball_detected:
                for d in range(len(measurement_timestamps)):
                    total_ball_radius_est.append(ball_radius_est[d])
                    total_ball_center_est.append([ball_center_est[d][0], ball_center_est[d][1]])
                    total_measurement_timestamps.append(measurement_timestamps[d])

                # Wwe can proceed with the trajectory prediction only if we have at least two estimates (it is the minimum number to fit a parabola)
                if len(total_ball_center_est) >= 2:
                    # We compute the time horizon for the trajectory prediction but subtracting to the length of the ground truth trajectory the time it has already been processed
                    t_horizon_ns = (config["pipeline"]["duration_trajectory_s"] - t_since_start) * 1e9
                    
                    # We compute the index of the closest timestamp in the Aria glasses odometry list to the current time
                    idx_aria_odom = min(
                        range(len(aria_glasses_odometry_timestamps)),
                        key=lambda k: abs(aria_glasses_odometry_timestamps[k] - trajectory_timestamps[index_traj]),
                    )

                    trajectory_3D_points_ARIA = transform_trajectory_to_ARIA_RGB(trajectory_3D_points,
                        aria_positions[np.abs(aria_pose_timestamps - trajectory_timestamps[index_traj]).argmin()],
                        aria_orientations[np.abs(aria_pose_timestamps - trajectory_timestamps[index_traj]).argmin()],
                        t_ARIA_RGB, R_ARIA_RGB)

                    trajectory_3D_points_ARIA = trajectory_3D_points_ARIA[trajectory_timestamps >= trajectory_timestamps[index_traj]]

                    img_rgb = provider.get_image_data_by_time_ns(rgb_stream_id, int(timestamps_aria_rgb[index]), time_domain, option)[0].to_numpy_array()
                    img_rgb = cv2.cvtColor(cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
                    img_rgb = draw_projected_points_on_image(img_rgb, trajectory_3D_points_ARIA, cam_calibration, color=(0, 255, 0), radius=5)

                    # If any point could not be projected, skip
                    if any(cam_calibration.project(pt) is None for pt in trajectory_3D_points_ARIA):
                        continue

                    # Compute ground truth points in world coordinates
                    trajectory_3D_points_W = transform_trajectory_to_W(trajectory_3D_points_ARIA, 
                                                              cam_calibration, transform_camera_device, 
                                                              aria_glasses_odometry[idx_aria_odom][1])

                    # If we have enough estimates, we can proceed with the trajectory prediction
                    if trajectory_positions_ODE is None:
                        # Run the monotonically constrained regression to refine the estimates
                        p_0, v_0, p_estimates, v_estimates = monotonically_constrained_regression(
                            total_ball_center_est,
                            total_ball_radius_est,
                            total_measurement_timestamps,
                            aria_glasses_odometry[idx_aria_odom][1], transform_camera_device,
                            R_RGB_DVS, t_RGB_DVS,
                            R_calib_comp, t_calib_comp,
                            num_IMU_samples, num_detections, (dt_IMU_ms * 1e-3),
                            config, has_rotated_calibration=has_rotated_calibration
                        )

                        # Run the learning-based trajectory prediction model (DCGM) for comparison with the ODE-based prediction
                        trajectory_positions_DCGM, _ = predict_trajectory_DCGM(
                            model,
                            p_estimates,
                            t_horizon_ns,
                            t_since_start,
                            dt_IMU_ms,
                            total_measurement_timestamps)

                        # Run ODE on the estimated measurements that we got from the ball detection module
                        t_viz_after_bounce = config["pipeline"]["propagate_traj_after_bounce"]
                        trajectory_positions_ODE, trajectory_est_timestamps = run_differential_equation_trajectory_prediction(
                                    p_0, v_0, dynamics_config, dt, t_viz_after_bounce)

                        # Bootstrap EKF initial state and compute trajectory
                        measurements = np.hstack([p_estimates[:-1], v_estimates])
                        position_states_EKF, velocity_states_EKF = bootstrap_initial_state_from_EKF(measurements, dt, dynamics_config, ekf_config)

                        p_0_ODE_EKF = position_states_EKF[-1]
                        v_0_ODE_EKF = np.mean(velocity_states_EKF, axis=0)
                        trajectory_positions_ODE_EKF, _ = run_differential_equation_trajectory_prediction(
                            p_0_ODE_EKF, v_0_ODE_EKF, dynamics_config, dt, t_viz_after_bounce)

                        # Extend ground truth trajectory if the current trajectory is not bouncing on the table
                        last_N_val = 40
                        trajectory_velocity_W = np.diff(trajectory_3D_points_W, axis=0) / dt
                        
                        v_0_GT = np.mean(trajectory_velocity_W[-last_N_val:], axis=0)
                        p_0_GT = trajectory_3D_points_W[-1]
                        trajectory_positions_propagated_GT, _ = run_differential_equation_trajectory_prediction(
                            p_0_GT, v_0_GT, dynamics_config, dt, 0.01, is_bouncing)
                        trajectory_positions_GT = np.vstack((downsample_3D_trajectory(trajectory_3D_points_W, len(trajectory_positions_ODE)), 
                                                             trajectory_positions_propagated_GT))

                        # Extend the DCGM trajectory if the current trajectory is not bouncing on the table
                        trajectory_positions_DCGM = savitzky_golay_filtering(trajectory_positions_DCGM)
                        trajectory_velocity_DCGM = np.diff(trajectory_positions_DCGM, axis=0) / dt
                        
                        v_0_DCGM = np.mean(trajectory_velocity_DCGM[-last_N_val:], axis=0)
                        p_0_DCGM = trajectory_positions_DCGM[-1]
                        trajectory_positions_propagated_DCGM, _ = run_differential_equation_trajectory_prediction(
                            p_0_DCGM, v_0_DCGM, dynamics_config, dt, t_viz_after_bounce)
                        trajectory_positions_DCGM = np.vstack((trajectory_positions_DCGM, trajectory_positions_propagated_DCGM))


                        """""""""""""""""""""""""""""""""""""""""""""""""""""
                        --------- PERFORMANCE EVALUATION METRICS ------------
                        """""""""""""""""""""""""""""""""""""""""""""""""""""

                        # Prepare all intersection arrays and names
                        intersection_configs = [
                            ("ODE", trajectory_positions_ODE),
                            ("ODE_EKF", trajectory_positions_ODE_EKF),
                            ("GT", trajectory_positions_GT),
                            ("DCGM", trajectory_positions_DCGM),
                        ]

                        intersections = {}
                        for name, arr in intersection_configs:
                            min_z_index = np.argmin(arr[:, 2])
                            intersections[name] = arr[min_z_index]

                        # Compute errors in a loop with descriptive names
                        error_pairs = [("ODE", "GT"), ("DCGM", "GT"), ("ODE_EKF", "GT")]
                        error_vars = ["error_ODE_vs_GT", "error_DCGM_vs_GT", "error_ODE_EKF_vs_GT"]

                        error_dict = {}
                        for (est, gt), err_name in zip(error_pairs, error_vars):
                            diff = intersections[est][:2] - intersections[gt][:2]
                            error_dict[err_name] = np.linalg.norm(diff)
                            
                        # Print error metrics in a loop for compactness
                        error_labels = ["ERROR [ODE vs. GT]", "ERROR [DCGM vs. GT]", "ERROR [ODE_EKF vs. GT]"]
                        error_values = [error_dict["error_ODE_vs_GT"], error_dict["error_DCGM_vs_GT"], error_dict["error_ODE_EKF_vs_GT"]]

                        for label, value in zip(error_labels, error_values):
                            print(f" {label}: {value}")

                        if config["io"]["save_evaluation_results"]:
                            # Save results with trajectory points, timestamps, and Aria glasses odometry
                            np.savez(os.path.join(args.path_input_sequence, "evaluation.npz"),
                                    points_GT = trajectory_3D_points_W,
                                    points_PRED = trajectory_positions_ODE,
                                    timestamps_PRED = trajectory_est_timestamps,
                                    aria_odometry_R = aria_glasses_odometry[idx_aria_odom][1],
                                    aria_position = aria_positions[np.abs(aria_pose_timestamps - trajectory_timestamps[index_traj]).argmin()],
                                    aria_orientation = aria_orientations[np.abs(aria_pose_timestamps - trajectory_timestamps[index_traj]).argmin()])

                            # Save performance statistics to a text file
                            with open(config["io"]["output_path_txt"], "a") as output_file:
                                # Prepare CSV row
                                event_str = ",".join(str(int(v)) for v in event_counts.values())
                                error_str = ",".join(str(v) for v in error_dict.values())
                                output_file.write(f"{total_measurement_timestamps[-1]},{event_str},{error_str}\n")
                                
                        """""""""""""""""""""""""""""""""""""""""""""""""""""
                        -----------------   VISUALIZATION   -----------------
                        """""""""""""""""""""""""""""""""""""""""""""""""""""
                        
                        if config["io"]["visualize_predicted_trajectory"]:
                            translation = aria_glasses_odometry[idx_aria_odom][0] - aria_glasses_odometry[idx_aria_odom - 1][0]
                            index = np.abs(timestamps_aria_rgb_dt - ((t0 - timestamps_events[0]) * 1e3)).argmin()
                            img_aria = provider.get_image_data_by_time_ns(rgb_stream_id, int(timestamps_aria_rgb[index]), time_domain, option)[0].to_numpy_array()

                            # Define a boundary circle around the intersection point
                            radius_circle = 0.25
                            theta = np.linspace(0, 2 * np.pi, 150)
                            circle_points = np.array([
                                intersections["GT"][0] + radius_circle * np.cos(theta),
                                intersections["GT"][1] + radius_circle * np.sin(theta),
                                dynamics_config["height_table"] * np.ones_like(theta) ])

                            # List of (points, color) to draw
                            viz_configs = [
                                (circle_points.T, (100, 100, 0)),
                                (trajectory_positions_ODE, (255, 0, 0)),
                                (trajectory_positions_ODE_EKF, (0, 0, 0)),
                                (trajectory_positions_GT, (0, 255, 0)),
                            ]
                            for points, color in viz_configs:
                                img_aria = draw_transformed_points_on_img(
                                    img_aria, points, R_RGB_DVS, t_RGB_DVS, aria_glasses_odometry[idx_aria_odom][1],
                                    translation, transform_camera_device, cam_calibration,
                                    R_calib_comp, t_calib_comp,
                                    K_events, D_events, color=color, radius=3, has_rotated_calibration=has_rotated_calibration,
                                )

                            # Legend position and display
                            rotated_img_aria = np.rot90(cv2.cvtColor(cv2.resize(img_aria, (704, 704)), cv2.COLOR_BGR2RGB), -1,)
                            cv2.imshow("Output trajectories (on ARIA frame [propagated])", rotated_img_aria)
                            cv2.waitKey(0)
                                         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("path_input_sequence", help="Path of the input sequence to test")

    args = parser.parse_args()

    with open(os.path.join(args.path_input_sequence, "config.yml"), "r") as f:
        config = yaml.safe_load(f)

    run_ball_trajectory_prediction(args, config)
