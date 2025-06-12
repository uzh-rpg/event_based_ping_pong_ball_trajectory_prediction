import numpy as np
import cv2
import dv_processing as dv


class NormalizedMeanTimestampImage:
    def __init__(self, height, width, K, rectified_points):
        self.height = height
        self.width = width
        self.K = K
        self.rectified_points = rectified_points
        self.normalized_mean_timestamp_image = np.zeros((height, width), dtype=np.float32)
        self.most_recent_event_ts = None

        self.normalized_mean_timestamp_image = np.zeros((height, width), dtype=np.float32)
        self.thresholded_mean_timestamp_image = np.zeros((height, width), dtype=np.uint8)
        self.over_threshold_image = np.zeros((height, width), dtype=np.float32)

        self.thresh_a_ = 0.1
        self.thresh_b_ = 0.05

        self.raw_image_filter_ = 1

        # Initialize row filter kernel
        raw_image_filter_kernel_r_ = np.zeros((1, 7), dtype=np.float32)
        raw_image_filter_kernel_r_[0, 0] = 2.0 / 301.0
        raw_image_filter_kernel_r_[0, 1] = 22.0 / 301.0
        raw_image_filter_kernel_r_[0, 2] = 97.0 / 301.0
        raw_image_filter_kernel_r_[0, 3] = 159.0 / 301.0
        raw_image_filter_kernel_r_[0, 4] = 97.0 / 301.0
        raw_image_filter_kernel_r_[0, 5] = 22.0 / 301.0
        raw_image_filter_kernel_r_[0, 6] = 2.0 / 301.0

        # Initialize column filter kernel
        raw_image_filter_kernel_c_ = np.zeros((7, 1), dtype=np.float32)
        raw_image_filter_kernel_c_[0, 0] = 2.0 / 301.0
        raw_image_filter_kernel_c_[1, 0] = 22.0 / 301.0
        raw_image_filter_kernel_c_[2, 0] = 97.0 / 301.0
        raw_image_filter_kernel_c_[3, 0] = 159.0 / 301.0
        raw_image_filter_kernel_c_[4, 0] = 97.0 / 301.0
        raw_image_filter_kernel_c_[5, 0] = 22.0 / 301.0
        raw_image_filter_kernel_c_[6, 0] = 2.0 / 301.0

        self.raw_image_filter_kernel_r_ = raw_image_filter_kernel_r_
        self.raw_image_filter_kernel_c_ = raw_image_filter_kernel_c_

        self.opening_iterations_ = 1
        self.closing_iterations_ = 1
        self.dilate_iterations_ = 0

        self.element_open_ = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.element_close_ = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.dilate_kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    def draw_normalized_mean_timestamp_image(self, events, w):

        event_store = dv.EventStore()
        self.normalized_mean_timestamp_image = np.zeros((self.height, self.width), dtype=np.float32)
        event_count = np.zeros((self.height, self.width), dtype=np.float32)

        I3 = np.eye(3)
        Sw = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        t0 = (events.timestamps()[0]) * 1e-6

        batch_size = 64
        batch_events = []

        index_batch = 0
        p_norm_cur = np.zeros(3, dtype=np.float32)
        p_norm_cur[2] = 1.0

        for ev in events:
            batch_events.append(ev)
            if index_batch >= batch_size - 1:

                for ev_ in batch_events:
                    dt = (ev_.timestamp()) * 1e-6 - t0

                    R = I3 - Sw * dt
                    KH_ref_cur = self.K @ R  # @ np.linalg.inv(self.K)

                    p = self.rectified_points[ev_.x() + ev_.y() * self.width]
                    p_norm_cur[:2] = p

                    p_ref_homogeneous = KH_ref_cur @ p_norm_cur
                    p_ref = p_ref_homogeneous[:2] / p_ref_homogeneous[2]

                    tl_x = int(np.floor(p_ref[0]))
                    tl_y = int(np.floor(p_ref[1]))

                    if 0 <= tl_x < self.width and 0 <= tl_y < self.height:

                        p_ref[0] - tl_x
                        p_ref[1] - tl_y

                        self.normalized_mean_timestamp_image[tl_y, tl_x] += dt
                        event_count[tl_y, tl_x] += 1

                        event_store.push_back(int(ev_.timestamp()), tl_x, tl_y, ev_.polarity())
                        # print(ev_.x(), ev_.y(), " VS ", tl_x, tl_y)

                batch_events = []
                index_batch = 0
            else:
                index_batch += 1

        # Normalize the mean timestamp image
        mean_rel_ts = 0.0
        count_rel_ts = 0
        for y in range(self.height):
            for x in range(self.width):
                if event_count[y, x] > 0:
                    self.normalized_mean_timestamp_image[y, x] /= event_count[y, x]
                    mean_rel_ts += self.normalized_mean_timestamp_image[y, x]
                    count_rel_ts += 1
                else:
                    assert self.normalized_mean_timestamp_image[y, x] == 0.0
                    self.normalized_mean_timestamp_image[y, x] = 0.0

        if count_rel_ts > 0:
            mean_rel_ts /= count_rel_ts

        deltaT = (events.timestamps()[-1] - events.timestamps()[0]) * 1e-6
        self.normalized_mean_timestamp_image = (self.normalized_mean_timestamp_image - mean_rel_ts) / deltaT

        # Update the most recent event timestamp
        self.most_recent_event_ts = events.timestamps()[-1] * 1e-6

        return event_store, event_count

    def mean_filter_image(self):
        if self.raw_image_filter_:
            # Apply separable filter (row filter and column filter)
            self.normalized_mean_timestamp_image = cv2.sepFilter2D(
                self.normalized_mean_timestamp_image,
                -1,
                self.raw_image_filter_kernel_r_,
                self.raw_image_filter_kernel_c_,
            )

    def threshold_mean_timestamp_image(self, w):
        # Compute the warp norm (similar to the Euclidean norm)
        warp_norm = np.sqrt(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)
        threshold = self.thresh_a_ * warp_norm + self.thresh_b_

        # Apply binary threshold to the image
        _, self.thresholded_mean_timestamp_image = cv2.threshold(self.normalized_mean_timestamp_image, threshold, 255, cv2.THRESH_BINARY)

        # Convert to 8-bit unsigned single-channel image
        self.thresholded_mean_timestamp_image = self.thresholded_mean_timestamp_image.astype(np.uint8)

    def filter_image(self):
        if self.opening_iterations_ > 0:
            # Apply morphological opening (erosion followed by dilation)
            self.thresholded_mean_timestamp_image = cv2.morphologyEx(
                self.thresholded_mean_timestamp_image,
                cv2.MORPH_OPEN,
                self.element_open_,
                iterations=self.opening_iterations_,
            )

        if self.closing_iterations_ > 0:
            # Apply morphological closing (dilation followed by erosion)
            self.thresholded_mean_timestamp_image = cv2.morphologyEx(
                self.thresholded_mean_timestamp_image,
                cv2.MORPH_CLOSE,
                self.element_close_,
                iterations=self.closing_iterations_,
            )

        if self.dilate_iterations_ > 0:
            # Apply dilation
            self.thresholded_mean_timestamp_image = cv2.dilate(
                self.thresholded_mean_timestamp_image,
                self.dilate_kernel_,
                iterations=self.dilate_iterations_,
            )

    def remove_below_threshold(self):

        self.over_threshold_image = self.normalized_mean_timestamp_image
        self.over_threshold_image[self.thresholded_mean_timestamp_image == 0] = 0

        return self.over_threshold_image
