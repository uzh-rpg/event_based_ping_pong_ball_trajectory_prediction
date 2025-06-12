import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
from utils.rigid_transformations import transform_point_RGB_to_DVS
from projectaria_tools.core.stream_id import RecordableTypeId



def reproject_ARIA_RGB_to_DVS(P1, img_point_ARIA, quat_12, t_12, K2, D2):
    img_point_ARIA_unproj = P1.unproject((int(img_point_ARIA[0]), int(img_point_ARIA[1])))

    img_point_DVS_unproj = transform_point_RGB_to_DVS(img_point_ARIA_unproj, t_12, Rot.from_quat(quat_12).as_matrix())
    img_point_DVS_unproj = img_point_DVS_unproj.reshape((3, 1))
    img_point_DVS_unproj[0] = -img_point_DVS_unproj[0]
    
    img_point_DVS, _ = cv2.projectPoints(img_point_DVS_unproj, np.zeros((3, 1)), np.zeros((3, 1)), K2, D2)
    img_point_DVS = img_point_DVS.ravel()[:2]

    return img_point_DVS

def project_points_DVS(point, K, D):
    img_point_DVS, _ = cv2.projectPoints(point, rvec=np.zeros(3), tvec=np.zeros(3), cameraMatrix=K, distCoeffs=D)
    img_point_DVS = img_point_DVS.squeeze()

    img_point_DVS_undist = cv2.undistortPoints(np.array([[[int(img_point_DVS[0]), int(img_point_DVS[1])]]], dtype=np.float32), K,  np.zeros((1, 4)))
    X, Y = img_point_DVS_undist[0][0]
    
    point_unproj = np.array([-X, Y, 1])

    return point_unproj, img_point_DVS


def get_aria_rgb_and_imu_streams(provider):
    # Initializes and returns RGB and IMU stream info, timestamps, and camera calibration from the Aria provider.
    options = provider.get_default_deliver_queued_options()
    options.deactivate_stream_all()

    rgb_stream_ids = options.get_stream_ids(RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS)
    left_imu_stream_id = provider.get_stream_id_from_label("imu-left")

    options.activate_stream(left_imu_stream_id)
    options.activate_stream(rgb_stream_ids[0])
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_ids[0])

    # Get the device and camera calibration
    device_calibration = provider.get_device_calibration()
    cam_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    return cam_calibration, device_calibration, rgb_stream_ids[0], rgb_stream_label


def compensate_rotated_calibration(img_point, R_calib_comp, t_calib_comp, P1, K2, D2):
    img_point_undist = cv2.undistortPoints(np.array([[[int(img_point[0]), int(img_point[1])]]], dtype=np.float32), K2, D2)
    X, Y = img_point_undist[0][0]

    point_unproj = np.array([X, Y, 1])
    point_unproj_rotated = np.linalg.inv(R_calib_comp) @ (point_unproj - t_calib_comp)
    image_point_rotated = P1.project(point_unproj_rotated)

    return image_point_rotated
