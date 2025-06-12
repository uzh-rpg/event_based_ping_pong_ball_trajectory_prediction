import copy
import numpy as np
import cv2
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
from utils.rigid_transformations import project_DVS_image_points_to_W


def predict_trajectory_DCGM(model, traj_3D_positions, t_horizon, t_from_start, dt_IMU, timestamps):

    dt_IMU_s = dt_IMU * 1e-3
    timestamps = np.array(timestamps)
    traj_3D_positions = np.array(traj_3D_positions).reshape(-1, 3)
    
    # Calculate the time horizon in seconds for prediction of the model
    t_horizon_model = (t_horizon + t_from_start * 1e9) * 1e-9
    N = int(t_horizon_model / dt_IMU_s)
    prediction_times = np.arange(0, N) * dt_IMU_s

    # Define previous observations and times used for prediction
    previous_obs = traj_3D_positions
    previous_times = (timestamps - timestamps[0])

    prediction_mean, prediction_cov = model.traj_dist(previous_times, previous_obs, prediction_times)

    return prediction_mean, prediction_cov


def fit_parabola(uv_points):
    # Fit a parabola to each coordinate of uv_points.
    uv_points = np.array(uv_points)
    for w in range(2):
        a, b, c = np.polyfit(range(len(uv_points)), uv_points[:, w], 2)
        uv_points[:, w] = a * np.arange(len(uv_points)) ** 2 + b * np.arange(len(uv_points)) + c
    return uv_points


def estimate_depths(avg_radii, fx, ball_radius):
    # Estimate depths Z from radii using camera intrinsics.
    return [fx * ball_radius / r for r in avg_radii]


def line(params, x):
    m, c = params
    return m * x + c


def loss_function(params, x, y):
    return np.sum((line(params, x) - y) ** 2)


def fit_monotonic_depth(Z_array, IMU_steps, dt, num_detections):
    # Fit a monotonic constraint to the depth array.
    v_low, v_up, v_init = -1.5, -6, -2.5
    z_lower = v_low * (IMU_steps * dt) / num_detections
    z_upper = v_up * (IMU_steps * dt) / num_detections
    z_initial_guess = v_init * (IMU_steps * dt) / num_detections

    constraints = [
        {"type": "ineq", "fun": lambda params: params[0] - z_lower},
        {"type": "ineq", "fun": lambda params: z_upper - params[0]},
    ]
    initial_guess = [z_initial_guess, np.mean(Z_array)]
    result = minimize(loss_function, initial_guess, args=(range(len(Z_array)), Z_array), constraints=constraints)
    m_opt, c_opt = result.x
    return [c_opt + i * m_opt for i in range(len(Z_array))]


def monotonically_constrained_regression(
    ball_center_measurements, ball_radius_measurements, timestamp_measurements, R_ARIA_w,
    transform_camera_device, R_RGB_DVS, t_RGB_DVS,
    R_calib_comp, t_calib_comp,
    num_IMU_samples, num_detections, dt_IMU,
    config, has_rotated_calibration=False):
    
    # Set parameters from configs
    calibration_config = config["calibration"]
    pipeline_config = config["pipeline"]
    ball_radius = pipeline_config.get("ball_radius", 0.02)
    fx = np.mean([calibration_config["K_events"][0][0], calibration_config["K_events"][1][1]])

    # Prepare data
    img_points = np.array(ball_center_measurements)
    radii = np.array(ball_radius_measurements)
    timestamps = np.array(timestamp_measurements)
    
    # Parabola fit for smoothing
    img_points = fit_parabola(img_points)

    # Depth estimation and monotonic fit
    Z = estimate_depths(radii, fx, ball_radius)
    Z = fit_monotonic_depth(np.array(Z), num_IMU_samples, dt_IMU, num_detections)

    # Reproject to world
    p_measurements = project_DVS_image_points_to_W(
        img_points, Z, calibration_config, transform_camera_device, R_ARIA_w, 
        R_RGB_DVS, t_RGB_DVS, has_rotated_calibration, R_calib_comp, t_calib_comp)

    # Velocity estimation
    dt = np.diff(timestamps) if len(timestamps) > 1 else np.array([1])
    v_measurements = np.diff(p_measurements, axis=0) / dt.reshape(-1, 1)
    p_0 = p_measurements[0] 
    v_0 = np.mean(v_measurements, axis=0)

    return p_0, v_0, p_measurements, v_measurements


def run_differential_equation_trajectory_prediction(initial_position, initial_velocity, dynamics_config, 
                                                    time_step, bounce_time_extension, has_already_bounced=False):
    # Unpack everything from dynamics_config
    mass = dynamics_config["mass"]
    drag_coefficient = dynamics_config["drag_coefficient"]
    magnus_coefficient = dynamics_config["magnus_coefficient"]
    air_density = dynamics_config["air_density"]
    radius = dynamics_config["radius"]
    table_height = dynamics_config["height_table"]
    g = np.array(dynamics_config["gravity"])
    bounce_factor = np.array(dynamics_config["bounce_factor"])
    angular_velocity = np.array(dynamics_config.get("omega", [0, 0, 0]))

    A = np.pi * radius**2  # Cross-sectional area of the ball (m^2)
    has_ball_bounced = False  # Flag to track if the ball has bounced

    # Initialize arrays for position, velocity, and acceleration
    position = [initial_position]
    velocity = [initial_velocity]
    acceleration = [np.zeros(3)]
    timestamps = [0]
    cum_timestamp = 0
    bounce_reached_time = 0

    if not has_already_bounced:
        # Keep simulating until the bounce condition is met
        while not has_ball_bounced:
            # Compute speed and unit velocity vector
            speed = np.linalg.norm(velocity[-1])
            if speed == 0:
                drag_force = np.array([0, 0, 0])
                magnus_force = np.array([0, 0, 0])
            else:
                unit_velocity = velocity[-1] / speed

                # Drag force
                drag_force = -0.5 * drag_coefficient * air_density * A * speed**2 * unit_velocity

                # Magnus force
                magnus_force = magnus_coefficient * air_density * A * speed * np.cross(angular_velocity, velocity[-1])

            # Total force
            total_force = mass * g + drag_force + magnus_force

            # Compute acceleration
            acceleration.append(total_force / mass)

            # Update velocity and position using Euler's method
            velocity.append(velocity[-1] + acceleration[-1] * time_step)
            position.append(position[-1] + velocity[-2] * time_step)

            # Update timestamp
            cum_timestamp += time_step
            timestamps.append(cum_timestamp)

            # Check if the ball hits the table and hasn't bounced yet
            if position[-1][2] <= table_height and velocity[-2][2] < 0 and not has_ball_bounced:
                velocity[-1][2] *= -1.0  # Reverse the z velocity to simulate bounce
                velocity[-1] = velocity[-1] * bounce_factor  # Apply bounce factor
                has_ball_bounced = True
                bounce_reached_time = cum_timestamp  # Mark the time when bounce occurred

    # After the ball has bounced, simulate for the additional time
    while cum_timestamp - bounce_reached_time < bounce_time_extension:
        # Compute speed and unit velocity vector
        speed = np.linalg.norm(velocity[-1])
        if speed == 0:
            drag_force = np.array([0, 0, 0])
            magnus_force = np.array([0, 0, 0])
        else:
            unit_velocity = velocity[-1] / speed

            # Drag force
            drag_force = -0.5 * drag_coefficient * air_density * A * speed**2 * unit_velocity

            # Magnus force
            magnus_force = magnus_coefficient * air_density * A * speed * np.cross(angular_velocity, velocity[-1])

        # Total force
        total_force = mass * g + drag_force + magnus_force

        # Compute acceleration
        acceleration.append(total_force / mass)

        # Update velocity and position using Euler's method
        velocity.append(velocity[-1] + acceleration[-1] * time_step)
        position.append(position[-1] + velocity[-2] * time_step)

        # Update timestamp
        cum_timestamp += time_step
        timestamps.append(cum_timestamp)

    return np.array(position), np.array(timestamps)
