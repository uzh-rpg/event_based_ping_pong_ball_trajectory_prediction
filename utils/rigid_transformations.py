import numpy as np
import copy
from scipy.spatial.transform import Rotation as Rot
import scipy.signal as signal


def transform_point(point_pos_W, pos_frame1_W, ori_frame1_W):
    R_frame1_W = Rot.from_quat(ori_frame1_W)

    # Construct the transformation matrix from W to A
    transformation_W_to_A = np.eye(4)
    transformation_W_to_A[:3, :3] = R_frame1_W.as_matrix()
    transformation_W_to_A[:3, 3] = pos_frame1_W
    transformation_A_to_W = np.linalg.inv(transformation_W_to_A)

    # Convert position of point B from frame W to frame A
    position_B_in_A_homogeneous = transformation_A_to_W @ np.append(point_pos_W, 1)
    position_B_in_A = position_B_in_A_homogeneous[:3]

    return position_B_in_A


def transform_traj_wrt_pose(points_world, position_body_w, orientation_body_w, position_cam_body, orientation_cam_body):

    N = points_world.shape[0]
    transformed_points = np.zeros((N, 3))

    for i in range(N):
        transformed_point = transform_point(points_world[i], position_body_w, orientation_body_w)
        transformed_point = transform_point(transformed_point, position_cam_body, orientation_cam_body)
        transformed_points[i] = transformed_point

    return transformed_points


def get_transformed_closest_points(
    points_world,
    point_timestamps,
    body_timestamps,
    position_body_w,
    orientation_body_w,
    position_cam_body,
    orientation_cam_body,
):
    T = len(body_timestamps)
    N = points_world.shape[0]
    transformed_points = np.zeros((T, 3))
    assert T == N

    for i in range(T):
        assert point_timestamps[i] == body_timestamps[i]

        transformed_point = transform_point(points_world[i], position_body_w[i], orientation_body_w[i])
        transformed_point = transform_point(transformed_point, position_cam_body, orientation_cam_body)
        transformed_points[i] = transformed_point

    return transformed_points


def transform_point_rgb_to_event(point_pos_W, pos_frame1_W, R_frame1_W_matrix):

    temp = copy.deepcopy(point_pos_W)
    temp[0] = point_pos_W[1]
    temp[1] = point_pos_W[0]
    point_pos_W = copy.deepcopy(temp)

    # Construct the transformation matrix from W to A
    transformation_W_to_A = np.eye(4)
    transformation_W_to_A[:3, :3] = R_frame1_W_matrix  # Rotation part
    transformation_W_to_A[:3, 3] = pos_frame1_W  # Translation part
    transformation_A_to_W = np.linalg.inv(transformation_W_to_A)

    # Convert position of point B from frame W to frame A
    position_B_in_A_homogeneous = transformation_A_to_W @ np.append(point_pos_W, 1)
    position_B_in_A = position_B_in_A_homogeneous[:3]  # Extract x, y, z coordinates

    return position_B_in_A


def transform_points_from_world_to_DVS_withTransl(
    points_world,
    R_quat,
    t,
    aria_orientation_WORLD,
    aria_translation_WORLD,
    transform_camera_device,
    cam_calibration,
    rotation_matrix_eyegaze,
    t_eyegaze,
    is_estimated_traj=False,
    is_new_dataset=False,
):
    points_DEVICE = np.linalg.inv(aria_orientation_WORLD) @ (points_world.T - aria_translation_WORLD.reshape(3, 1))
    points_ARIA = (transform_camera_device[0:3, 0:3] @ points_DEVICE).T + transform_camera_device[0:3, 3].reshape(1, 3)
    points_DVS = []
    for i in range(len(points_ARIA)):

        temp = copy.deepcopy(points_ARIA[i])
        temp[0] = points_ARIA[i][1]
        temp[1] = points_ARIA[i][0]

        transformation_W_to_A = np.eye(4)
        transformation_W_to_A[:3, :3] = Rot.from_quat(R_quat).as_matrix()  # Rotation part
        transformation_W_to_A[:3, 3] = t
        transformation_A_to_W = np.linalg.inv(transformation_W_to_A)
        position_B_in_A_homogeneous = transformation_A_to_W @ np.append(temp, 1)
        position_B_in_A = position_B_in_A_homogeneous[:3]
        point_in_camera_1 = position_B_in_A.reshape((3, 1))
        if not is_estimated_traj:
            point_in_camera_1[0] = -point_in_camera_1[0]

        if is_new_dataset:
            point_in_camera_1 = rotation_matrix_eyegaze @ points_ARIA[i] + t_eyegaze

        points_DVS.append(point_in_camera_1.ravel())

    return np.array(points_DVS), points_ARIA


def transform_position_to_frame_2(point_pos_W, pos_frame1_W, ori_frame1_W):

    R_frame1_W = Rot.from_quat(ori_frame1_W)

    # Construct the transformation matrix from W to A
    transformation_W_to_A = np.eye(4)
    transformation_W_to_A[:3, :3] = R_frame1_W.as_matrix()  # Rotation part
    transformation_W_to_A[:3, 3] = pos_frame1_W  # Translation part
    transformation_A_to_W = np.linalg.inv(transformation_W_to_A)

    # Convert position of point B from frame W to frame A
    position_B_in_A_homogeneous = transformation_A_to_W @ np.append(point_pos_W, 1)
    position_B_in_A = position_B_in_A_homogeneous[:3]  # Extract x, y, z coordinates

    return position_B_in_A


def compute_rmse(traj1, traj2):
    """
    Compute the RMSE between two 3D trajectories.

    Parameters:
        traj1 (numpy.ndarray): First trajectory, shape (N, 3).
        traj2 (numpy.ndarray): Second trajectory, shape (N, 3).

    Returns:
        float: RMSE value.
    """
    if traj1.shape != traj2.shape:
        raise ValueError("Trajectories must have the same shape.")

    # Compute the squared differences
    squared_diff = np.sum((traj1 - traj2) ** 2, axis=1)

    # Mean of the squared differences
    mse = np.mean(squared_diff)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    return rmse


def downsample_3d_array(data, target_size):
    """
    Downsample a 3D array to the target size using linear interpolation.

    Parameters:
        data (numpy.ndarray): Original 3D array of shape (N, 3).
        target_size (int): Desired number of rows (M).

    Returns:
        numpy.ndarray: Downsampled 3D array of shape (M, 3).
    """
    original_size = data.shape[0]

    # Define original and target indices
    original_indices = np.linspace(0, original_size - 1, original_size)
    target_indices = np.linspace(0, original_size - 1, target_size)

    # Interpolate each dimension
    downsampled_data = np.array([np.interp(target_indices, original_indices, data[:, dim]) for dim in range(data.shape[1])]).T

    return downsampled_data


def savitzky_golay_filtering(data, window_size=300, poly_order=2):
    smoothed_data = np.array([signal.savgol_filter(data[:, i], window_size, poly_order) for i in range(data.shape[1])]).T
    return smoothed_data


def get_transformed_trajectory(
    points_world,
    point_timestamps,
    body_timestamps,
    position_body_w,
    orientation_body_w,
    position_cam_body,
    orientation_cam_body,
):
    T = len(body_timestamps)
    N = points_world.shape[0]
    transformed_trajectories = []
    trajectories_timestamps = []

    t_horizon = 50

    assert T == N
    for i in range(T):
        transformed_trajectory = []
        trajectory_timestamps = []
        for j in range(t_horizon):
            if (i + j) < points_world.shape[0]:
                transformed_point = transform_position_to_frame_2(points_world[i + j], position_body_w[i], orientation_body_w[i])
                transformed_point = transform_position_to_frame_2(transformed_point, position_cam_body, orientation_cam_body)
                transformed_trajectory.append(transformed_point)
                trajectory_timestamps.append(point_timestamps[i + j])

        transformed_trajectories.append(transformed_trajectory)
        trajectories_timestamps.append(trajectory_timestamps)

    return transformed_trajectories, trajectories_timestamps


def get_transformed_trajectory(
    points_world,
    point_timestamps,
    body_timestamps,
    position_body_w,
    orientation_body_w,
    position_cam_body,
    orientation_cam_body,
):
    T = len(body_timestamps)
    N = points_world.shape[0]
    transformed_trajectories = []
    trajectories_timestamps = []

    assert T == N
    for i in range(T):
        transformed_trajectory = []
        trajectory_timestamps = []
        for j in range(points_world.shape[0]):
            transformed_point = transform_position_to_frame_2(points_world[j], position_body_w[i], orientation_body_w[i])
            transformed_point = transform_position_to_frame_2(transformed_point, position_cam_body, orientation_cam_body)
            transformed_trajectory.append(transformed_point)
            trajectory_timestamps.append(point_timestamps[j])

        transformed_trajectories.append(transformed_trajectory)
        trajectories_timestamps.append(trajectory_timestamps)

    return transformed_trajectories, trajectories_timestamps


def get_transformed_trajectory_approx(
    points_world,
    point_timestamps,
    body_timestamps,
    position_body_w,
    orientation_body_w,
    position_cam_body,
    orientation_cam_body,
):
    T = len(body_timestamps)
    N = points_world.shape[0]
    transformed_trajectories = []
    t_horizon = 100

    for i in range(N):
        transformed_trajectory = []
        idx_closest = (np.abs(np.array(body_timestamps) - int(point_timestamps[i]))).argmin()

        for j in range(t_horizon):
            if (i + j) < points_world.shape[0]:
                transformed_point = transform_position_to_frame_2(
                    points_world[i + j],
                    position_body_w[idx_closest],
                    orientation_body_w[idx_closest],
                )
                transformed_point = transform_position_to_frame_2(transformed_point, position_cam_body, orientation_cam_body)
                transformed_trajectory.append(transformed_point)

        transformed_trajectories.append(transformed_trajectory)
    return transformed_trajectories
