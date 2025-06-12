import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.rigid_transformations import transform_trajectory_W_to_DVS, transform_trajectory_DVS_to_RGB
from utils.camera import project_points_DVS, compensate_rotated_calibration

def draw_projected_points_on_image(img, points_3d, cam_calibration, color=(0, 255, 0), radius=5):
    # Draw projected 3D points on an image.
    for pt in points_3d:
        uv = cam_calibration.project(pt)
        if uv is not None:
            cv2.circle(img, (int(uv[0]), int(uv[1])), radius, color, -1)
        else:
            break
    return img

def draw_transformed_points_on_img(
    img, points_world, R_RGB_DVS, t_RGB_DVS, aria_rotation, aria_translation, 
    T_RGB_DEVICE, cam_calibration, R_calib_comp, t_calib_comp, 
    K_events, D_events, color, radius=3, has_rotated_calibration=False):

    # Transform world points, project to ARIA image, and draw them.
    points_DVS, _ = transform_trajectory_W_to_DVS(
        points_world, R_RGB_DVS, t_RGB_DVS, aria_rotation, aria_translation, T_RGB_DEVICE, 
        R_calib_comp, t_calib_comp, has_rotated_calibration=has_rotated_calibration)
    
    # Project and draw each point on the image.
    for point in points_DVS:
        point_unprojected, point_projected_2D = project_points_DVS(point.reshape(1, 3), K_events, D_events)
        point_ARIA_RGB = transform_trajectory_DVS_to_RGB(point_unprojected, R_RGB_DVS, t_RGB_DVS)
        img_point_ARIA_RGB = cam_calibration.project(point_ARIA_RGB)
        
        if R_calib_comp is not None:
            img_point_ARIA_RGB = compensate_rotated_calibration(point_projected_2D, R_calib_comp, t_calib_comp, cam_calibration, K_events, D_events)
            
        if img_point_ARIA_RGB is not None:
            img_point_ARIA_RGB = img_point_ARIA_RGB.ravel()[:2]
            cv2.circle(img, (int(img_point_ARIA_RGB[0]), int(img_point_ARIA_RGB[1])), radius, color, -1)

    return img

def plot_mic_signals(provider):
    # Plot microphone signals from the provider.
    mic_stream_id = provider.get_stream_id_from_label("mic")
    num_mic_samples = provider.get_num_data(mic_stream_id)
    timestamps = []
    audio = [[] for _ in range(0, 7)]
    
    # Iterate through microphone samples and collect audio data and timestamps.
    for index in range(0, num_mic_samples):
        audio_data_i = provider.get_audio_data_by_index(mic_stream_id, index)
        audio_signal_block = audio_data_i[0].data
        timestamps_block = [t * 1e-9 for t in audio_data_i[1].capture_timestamps_ns]
        timestamps += timestamps_block
        for c in range(0, 7):
            audio[c] += audio_signal_block[c::7]

    # Plot the audio signals.
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    fig.suptitle(f"Microphone signal")
    for c in range(0, 7):
        plt.plot(timestamps, audio[c], "-", label=f"channel {c}")
    axes.legend(loc="upper left")
    axes.grid("on")
    axes.set_xlabel("timestamps (s)")
    axes.set_ylabel("audio readout")
    plt.show()
