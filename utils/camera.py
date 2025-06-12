import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2
from utils.rigid_transformations import transform_point_rgb_to_event


def reproject_pt_in_event_frame(P1, img_event, img_point1, R_quat_12, t_12, K2, dist_coeffs2):
    eyegaze_unprojected = P1.unproject((int(img_point1[0]), int(img_point1[1])))

    point_in_camera_2 = transform_point_rgb_to_event(eyegaze_unprojected, t_12, Rot.from_quat(R_quat_12).as_matrix())
    point_in_camera_2 = point_in_camera_2.reshape((3, 1))
    point_in_camera_2[0] = -point_in_camera_2[0]
    image_point_2, _ = cv2.projectPoints(point_in_camera_2, np.zeros((3, 1)), np.zeros((3, 1)), K2, dist_coeffs2)
    point_ev_repr = image_point_2.ravel()[:2]

    return point_ev_repr
