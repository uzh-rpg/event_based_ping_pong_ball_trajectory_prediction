import numpy as np
import copy

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

import cv2

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from scipy.spatial.transform import Rotation as Rot


def reproject_2d_to_3d(uv_points, Z, K):
    uv_homogeneous = np.hstack([uv_points, np.ones((uv_points.shape[0], 1))])
    K_inv = np.linalg.inv(K)

    xyz = []
    for i in range(len(Z)):
        xyz.append(Z[i] * ((K_inv @ uv_homogeneous.T).T)[i, :])
    xyz = np.array(xyz)

    return xyz


def upsample_positions(positions, t_original, t_target):
    """
    Upsamples a 3D array of positions (x, y, z) from size N to size M using quadratic interpolation.

    Parameters:
    - positions (np.ndarray): Input array of shape (N, 3).
    - target_size (int): Desired output size M (M > N).

    Returns:
    - np.ndarray: Upsampled array of shape (M, 3).
    """
    # Original size
    positions.shape[0]

    # Interpolating each dimension (x, y, z) separately
    upsampled_positions = np.zeros((t_target.shape[0], 3))
    for i in range(3):  # Loop over dimensions
        # Fit a quadratic polynomial (degree=2) to the original data
        coefficients = np.polyfit(t_original, positions[:, i], deg=1)
        # Evaluate the polynomial at the target time scale
        upsampled_positions[:, i] = np.polyval(coefficients, t_target)

    return upsampled_positions


def line(params, x):
    m, c = params
    return m * x + c


def loss_function(params, x, y):
    return np.sum((line(params, x) - y) ** 2)


def predict_trajectory_DCGM(model, traj_3D_positions, t_horizon, cum_dt, timestamps):
    traj_3D_positions = np.array(traj_3D_positions).reshape(-1, 3)

    dt_model = 0.00125353
    t_horizon_model = (t_horizon + cum_dt * 1e9) * 1e-9
    steps = int(t_horizon_model / dt_model)
    pred_times = np.arange(0, steps) * dt_model

    t_original = timestamps
    t_original -= t_original[0]
    t_target = np.linspace(0, t_original[-1], int(t_original[-1] / dt_model) - 1)
    prev_obs = upsample_positions(traj_3D_positions, t_original, t_target)
    prev_times = copy.deepcopy(t_target)

    prev_obs = traj_3D_positions
    prev_times = t_original

    pred_mean, pred_cov = model.traj_dist(prev_times, prev_obs, pred_times)

    return pred_mean, pred_cov

def fit_parabola(uv_points):
    """Fit a parabola to each coordinate of uv_points."""
    uv_points = np.array(uv_points)
    for w in range(2):
        a, b, c = np.polyfit(range(len(uv_points)), uv_points[:, w], 2)
        uv_points[:, w] = a * np.arange(len(uv_points)) ** 2 + b * np.arange(len(uv_points)) + c
    return uv_points

def estimate_depths(avg_radii, fx, ball_radius):
    """Estimate depths Z from radii using camera intrinsics."""
    return [fx * ball_radius / r for r in avg_radii]

def fit_monotonic_depth(Z_array, step, num_int_frame_dbscan):
    """Fit a monotonic constraint to the depth array."""
    v_low, v_up, v_init = -1.5, -6, -2.5
    z_lower = v_low * (step * 0.00125) / num_int_frame_dbscan
    z_upper = v_up * (step * 0.00125) / num_int_frame_dbscan
    z_initial_guess = v_init * (step * 0.00125) / num_int_frame_dbscan

    constraints = [
        {"type": "ineq", "fun": lambda params: params[0] - z_lower},
        {"type": "ineq", "fun": lambda params: z_upper - params[0]},
    ]
    initial_guess = [z_initial_guess, np.mean(Z_array)]
    result = minimize(loss_function, initial_guess, args=(range(len(Z_array)), Z_array), constraints=constraints)
    m_opt, c_opt = result.x
    return [c_opt + i * m_opt for i in range(len(Z_array))]

def reproject_points_to_world(
    uv_points, Z, calibration_config, transform_camera_device, aria_w_orientation, R_quat, t, is_new_dataset, rotation_matrix_eyegaze, t_eyegaze
):
    """Reproject 2D points to world coordinates."""
    traj_3D_positions_WORLD = []
    for a in range(len(uv_points)):
        undistorted_points = cv2.undistortPoints(
            np.array([[[int(uv_points[a][0]), int(uv_points[a][1])]]], dtype=np.float32),
            np.array(calibration_config["K_events"]),
            np.array(calibration_config["D_events"]),
        )
        X, Y = undistorted_points[0][0]
        point_unprojected = Z[a] * np.array([-X, Y, 1])
        transformation_W_to_A = np.eye(4)
        transformation_W_to_A[:3, :3] = np.linalg.inv(Rot.from_quat(R_quat).as_matrix())
        transformation_W_to_A[:3, 3] = -t
        transformation_A_to_W = np.linalg.inv(transformation_W_to_A)
        position_B_in_A_homogeneous = transformation_A_to_W @ np.append(point_unprojected, 1)
        position_B_in_A = position_B_in_A_homogeneous[:3]
        point_in_camera_1 = position_B_in_A.reshape((3, 1))
        temp = copy.deepcopy(point_in_camera_1)
        temp[0] = point_in_camera_1[1]
        temp[1] = point_in_camera_1[0]
        if is_new_dataset and rotation_matrix_eyegaze is not None and t_eyegaze is not None:
            test_1 = Z[a] * np.array([X, Y, 1])
            temp = np.linalg.inv(rotation_matrix_eyegaze) @ (test_1 - t_eyegaze)
        point_device = np.linalg.inv(transform_camera_device[0:3, 0:3]) @ (temp.reshape(3, 1) - transform_camera_device[0:3, 3].reshape(3, 1))
        point_world = aria_w_orientation @ point_device.reshape(3, 1)
        traj_3D_positions_WORLD.append(point_world)
    return np.array(traj_3D_positions_WORLD)[:, :, 0]

def monotonically_constrained_regression(
    point_centers,
    avg_radii,
    timestamps_interm,
    aria_w_orientation,
    transform_camera_device,
    R_quat,
    t,
    rotation_matrix_eyegaze,
    t_eyegaze,
    step,
    num_int_frame_dbscan,
    config,
    is_new_dataset=False):
    
    # Configs
    calibration_config = config["calibration"]
    pipeline_config = config["pipeline"]
    ball_radius = pipeline_config.get("ball_radius", 0.02)
    fx = np.mean([calibration_config["K_events"][0][0], calibration_config["K_events"][1][1]])

    # Prepare data
    uv_points = np.array(point_centers)
    avg_radii = np.array(avg_radii)
    timestamps = np.array(timestamps_interm)
    
    # Parabola fit for smoothing
    uv_points = fit_parabola(uv_points)

    # Depth estimation and monotonic fit
    Z = estimate_depths(avg_radii, fx, ball_radius)
    Z = fit_monotonic_depth(np.array(Z), step, num_int_frame_dbscan)

    # Reproject to world
    traj_3D_positions = reproject_points_to_world(
        uv_points, Z, calibration_config, transform_camera_device, aria_w_orientation, R_quat, t, is_new_dataset, rotation_matrix_eyegaze, t_eyegaze
    )

    # Velocity estimation
    dt_arr = np.diff(timestamps) if len(timestamps) > 1 else np.array([1])
    traj_3D_velocities_array = np.diff(traj_3D_positions, axis=0) / dt_arr.reshape(-1, 1)
    initial_p = traj_3D_positions[-1] if traj_3D_positions.shape[0] > 20 else traj_3D_positions[0]
    initial_v = np.mean(traj_3D_velocities_array[-20:], axis=0) if traj_3D_velocities_array.shape[0] > 20 else np.mean(traj_3D_velocities_array, axis=0)

    return initial_p, initial_v, traj_3D_positions, traj_3D_velocities_array


# def compute_trajectory_from_position(
#     initial_position,
#     initial_velocity,
#     dynamics_config,
#     time_step,
#     total_time,
#     is_gt=False,
# ):
    
#     # Unpack everything from dynamics_config
#     mass = dynamics_config["mass"]
#     drag_coefficient = dynamics_config["drag_coefficient"]
#     magnus_coefficient = dynamics_config["magnus_coefficient"]
#     air_density = dynamics_config["air_density"]
#     radius = dynamics_config["radius"]
#     table_height = dynamics_config["height_table"]
#     g = np.array(dynamics_config["gravity"])
#     bounce_factor = np.array(dynamics_config["bounce_factor"])
#     angular_velocity = np.array(dynamics_config.get("omega", [0, 0, 0]))
    
#     # Constants
#     A = np.pi * radius**2
    
#     if is_gt:
#         bounce_factor = np.array([1, 1, 1])
#     else:
#         bounce_factor = np.array([0.4, 0.4, 0.4])

#     has_ball_bounced = False
#     table_height = table_height

#     # Time steps
#     num_steps = int(total_time / time_step)

#     # Initialize arrays for position, velocity, and acceleration
#     position = np.zeros((num_steps, 3))
#     velocity = np.zeros((num_steps, 3))
#     acceleration = np.zeros((num_steps, 3))
#     timestamps = np.zeros((num_steps,))
#     cum_timestamp = 0

#     # Set initial conditions
#     position[0] = initial_position
#     velocity[0] = initial_velocity

#     for i in range(1, num_steps):
#         # Compute speed and unit velocity vector
#         speed = np.linalg.norm(velocity[i - 1])
#         if speed == 0:
#             drag_force = np.array([0, 0, 0])
#             magnus_force = np.array([0, 0, 0])
#         else:
#             unit_velocity = velocity[i - 1] / speed

#             # Drag force
#             drag_force = -0.5 * drag_coefficient * air_density * A * speed**2 * unit_velocity

#             # Magnus force
#             magnus_force = magnus_coefficient * air_density * A * speed * np.cross(angular_velocity, velocity[i - 1])

#         # Total force
#         total_force = mass * g + drag_force + magnus_force

#         # Compute acceleration
#         acceleration[i] = total_force / mass

#         # Update velocity and position using Euler's method
#         velocity[i] = velocity[i - 1] + acceleration[i] * time_step
#         position[i] = position[i - 1] + velocity[i - 1] * time_step

#         if position[i][2] <= table_height and velocity[i - 1][2] < 0 and not has_ball_bounced:
#             velocity[i][2] *= -1.0
#             velocity[i] = velocity[i] * bounce_factor
#             has_ball_bounced = True

#         cum_timestamp += time_step
#         timestamps[i] = cum_timestamp

#     return position, timestamps


def compute_trajectory_from_position_untilBOUNCE(
    initial_position,
    initial_velocity,
    dynamics_config,
    time_step,
    # total_time,
    bounce_time_extension,
    has_already_bounced=False,
):

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

    # bounce_time_extension = 0.12  # Time to extend the simulation after bounce (seconds)
        
    A = np.pi * radius**2  # Cross-sectional area of the ball (m^2)

    has_ball_bounced = False  # Flag to track if the ball has bounced

    # Initialize arrays for position, velocity, and acceleration
    position = [initial_position]  # Start with initial position
    velocity = [initial_velocity]  # Start with initial velocity
    acceleration = [np.zeros(3)]  # Start with initial acceleration
    timestamps = [0]  # Start with timestamp 0
    cum_timestamp = 0
    bounce_reached_time = 0  # Time when the bounce occurred

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


    # After the ball has bounced, simulate for the additional 0.2 seconds
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
