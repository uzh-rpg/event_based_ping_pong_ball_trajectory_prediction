import numpy as np
import copy
import cv2
from scipy.spatial.transform import Rotation as Rot
import scipy.signal as signal


def transform_point(point_W, t_A_W, quat_A_W):
    R_A_W = Rot.from_quat(quat_A_W)

    # Construct the transformation matrix from W to A
    T_A_W = np.eye(4)
    T_A_W[:3, :3] = R_A_W.as_matrix()
    T_A_W[:3, 3] = t_A_W
    T_W_A = np.linalg.inv(T_A_W)

    # Convert position of point B from frame W to frame A
    point_A_hom = T_W_A @ np.append(point_W, 1)
    point_A = point_A_hom[:3]

    return point_A


def transform_trajectory_to_ARIA_RGB(points_W, position_ARIA_W, orientation_ARIA_W, position_RGB_ARIA, orientation_RGB_ARIA):

    N = points_W.shape[0]
    points_ARIA_RGB = np.zeros((N, 3))

    for i in range(N):
        point_ARIA = transform_point(points_W[i], position_ARIA_W, orientation_ARIA_W)
        point_ARIA_RGB = transform_point(point_ARIA, position_RGB_ARIA, orientation_RGB_ARIA)
        points_ARIA_RGB[i] = point_ARIA_RGB

    return points_ARIA_RGB


def transform_point_RGB_to_DVS(point_RGB, position_RGB_DVS, R_RGB_DVS):

    point_RGB_copy = copy.deepcopy(point_RGB)
    point_RGB_copy[0] = point_RGB[1]
    point_RGB_copy[1] = point_RGB[0]

    # Construct the transformation matrix from W to A
    T_DVS_RGB = np.eye(4)
    T_DVS_RGB[:3, :3] = R_RGB_DVS
    T_DVS_RGB[:3, 3] = position_RGB_DVS 
    T_RGB_DVS = np.linalg.inv(T_DVS_RGB)

    # Convert position of point B from frame W to frame A
    point_DVS_hom = T_RGB_DVS @ np.append(point_RGB_copy, 1)
    point_DVS = point_DVS_hom[:3]

    return point_DVS


def transform_trajectory_to_W(trajectory, cam_calibration, T_RGB_DEVICE, R_odometry):
    # Compute ground truth points in world coordinates.
    points_W = []
    
    for point in trajectory:
        img_point_ARIA = cam_calibration.project(point)
        
        img_point_ARIA_rotated = copy.deepcopy(img_point_ARIA)
        img_point_ARIA_rotated[0] = img_point_ARIA[1]
        img_point_ARIA_rotated[1] = cam_calibration.get_image_size()[0] - img_point_ARIA[0]
        
        point_ARIA = cam_calibration.unproject((int(img_point_ARIA_rotated[0]), int(img_point_ARIA_rotated[1]))) * point[-1]
        point_DEVICE = np.linalg.inv(T_RGB_DEVICE[0:3, 0:3]) @ (
            point_ARIA.reshape(3, 1) - T_RGB_DEVICE[0:3, 3].reshape(3, 1))
        
        point_W = R_odometry @ point_DEVICE.reshape(3, 1)
        points_W.append(point_W)
        
    return np.array(points_W)[:, :, 0]




def transform_trajectory_DVS_to_RGB(point_DVS, R_RGB_DVS, t_RGB_DVS):
    # Transform a point from DVS coordinates to RGB coordinates using the provided rotation and translation.
    T_RGB_DVS = np.eye(4)
    T_RGB_DVS[:3, :3] = np.linalg.inv(Rot.from_quat(R_RGB_DVS).as_matrix()) 
    T_RGB_DVS[:3, 3] = -t_RGB_DVS
    
    T_DVS_RGB = np.linalg.inv(T_RGB_DVS)
    point_RGB_hom = T_DVS_RGB @ np.append(point_DVS, 1)
    point_RGB = point_RGB_hom[:3]
    point_RGB = point_RGB.reshape((3, 1))

    point_RGB_rotated = copy.deepcopy(point_RGB)
    point_RGB_rotated[0] = point_RGB[1]
    point_RGB_rotated[1] = point_RGB[0]

    return point_RGB_rotated


def transform_trajectory_W_to_DVS(points_W, R_RGB_DVS, t_RGB_DVS, R_ARIA_W, t_ARIA_W,
    T_RGB_DEVICE, R_calib_comp, t_calib_comp, has_rotated_calibration=False):
    
    points_DEVICE = np.linalg.inv(R_ARIA_W) @ (points_W.T - t_ARIA_W.reshape(3, 1))
    points_ARIA = (T_RGB_DEVICE[0:3, 0:3] @ points_DEVICE).T + T_RGB_DEVICE[0:3, 3].reshape(1, 3)
    points_DVS = []
    for i in range(len(points_ARIA)):

        points_ARIA_copy = copy.deepcopy(points_ARIA[i])
        points_ARIA_copy[0] = points_ARIA[i][1]
        points_ARIA_copy[1] = points_ARIA[i][0]

        T_DVS_RGB = np.eye(4)
        T_DVS_RGB[:3, :3] = Rot.from_quat(R_RGB_DVS).as_matrix()
        T_DVS_RGB[:3, 3] = t_RGB_DVS
        T_RGB_DVS = np.linalg.inv(T_DVS_RGB)
        point_DVS_hom = T_RGB_DVS @ np.append(points_ARIA_copy, 1)
        point_DVS = point_DVS_hom[:3]
        point_DVS = point_DVS.reshape((3, 1))
        point_DVS[0] = -point_DVS[0]

        if has_rotated_calibration:
            point_DVS = R_calib_comp @ points_ARIA[i] + t_calib_comp

        points_DVS.append(point_DVS.ravel())

    return np.array(points_DVS), points_ARIA


def project_DVS_image_points_to_W(img_points, Z, calibration_config, T_RGB_DEVICE, R_ARIA_W, 
                              R_RGB_DVS, t_RGB_DVS, has_rotated_calibration, R_calib_comp, t_calib_comp):
    # Reproject 2D points to world coordinates.
    points_W = []
    
    for k in range(len(img_points)):
        # Undistort the image coordinates using K and D
        img_points_undist = cv2.undistortPoints(
            np.array([[[int(img_points[k][0]), int(img_points[k][1])]]], dtype=np.float32),
            np.array(calibration_config["K_events"]),
            np.array(calibration_config["D_events"]))
        X, Y = img_points_undist[0][0]
        point_unprojected = Z[k] * np.array([-X, Y, 1])
        
        # Transform from DVS to ARIA-RGB frame
        T_RGB_DVS = np.eye(4)
        T_RGB_DVS[:3, :3] = np.linalg.inv(Rot.from_quat(R_RGB_DVS).as_matrix())
        T_RGB_DVS[:3, 3] = -t_RGB_DVS
        T_DVS_RGB = np.linalg.inv(T_RGB_DVS)
        
        point_ARIA_hom = T_DVS_RGB @ np.append(point_unprojected, 1)
        point_ARIA = point_ARIA_hom[:3]
        point_ARIA = point_ARIA.reshape((3, 1))
        
        point_ARIA_copy = copy.deepcopy(point_ARIA)
        point_ARIA_copy[0] = point_ARIA[1]
        point_ARIA_copy[1] = point_ARIA[0]
        
        if has_rotated_calibration and R_calib_comp is not None and t_calib_comp is not None:
            point_DVS = Z[k] * np.array([X, Y, 1])
            point_ARIA_copy = np.linalg.inv(R_calib_comp) @ (point_DVS - t_calib_comp)
            
        point_DEVICE = np.linalg.inv(T_RGB_DEVICE[0:3, 0:3]) @ (point_ARIA_copy.reshape(3, 1) - T_RGB_DEVICE[0:3, 3].reshape(3, 1))
        point_W = R_ARIA_W @ point_DEVICE.reshape(3, 1)
        points_W.append(point_W)
        
    return np.array(points_W)[:, :, 0]



def downsample_3D_trajectory(trajectory, target_size):
    """
    Downsample a 3D trajectory to the target size using linear interpolation.

    Returns:
        numpy.ndarray: Downsampled 3D trajectory of shape (M, 3).
    """
    original_size = trajectory.shape[0]

    # Define original and target indices
    original_indices = np.linspace(0, original_size - 1, original_size)
    target_indices = np.linspace(0, original_size - 1, target_size)

    # Interpolate each dimension
    downsampled_trajectory = np.array([np.interp(target_indices, original_indices, trajectory[:, dim]) for dim in range(trajectory.shape[1])]).T

    return downsampled_trajectory


def savitzky_golay_filtering(data, window_size=300, poly_order=2):
    # Apply Savitzky-Golay filtering to smooth the data.
    smoothed_data = np.array([
        signal.savgol_filter(data[:, i], window_size, poly_order) 
        for i in range(data.shape[1])]
                             ).T
    return smoothed_data

