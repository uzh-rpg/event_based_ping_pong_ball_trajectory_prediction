import numpy as np
import dv_processing as dv
import copy

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

def get_filtered_events_store(blurred_mask, input_events, visualizer):
    # Normalize the blurred mask to the range [0, 255]
    blurred_mask = 255 * ((blurred_mask - np.min(blurred_mask)) / (np.max(blurred_mask) - np.min(blurred_mask)))

    # Generate the frame using the visualizer and apply the blurred mask
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
    
    # Iterate through the filtered event coordinates and find matches in the original events
    for _, point in enumerate(filtered_event_coords):
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

    # Sort the filtered events by timestamp and push them to the filtered event store
    events_filt = np.array(events_filt)
    events_filt = events_filt[events_filt[:, 0].argsort()]

    for ev in events_filt:
        filtered_event_store.push_back(ev[0], ev[1], ev[2], bool(ev[3]))

    num_events_dyn_obj = len(filtered_event_store.timestamps())

    return filtered_event_store, num_events_dyn_obj


def crop_image_around_point(img_shape, center, crop_size):
    # Extract height and width from the image shape
    h, w = img_shape[:2]
    y, x = int(center[1]), int(center[0])
    
    # Ensure the crop does not go out of the image bounds
    y1 = max(0, y - crop_size)
    y2 = min(h, y + crop_size)
    x1 = max(0, x - crop_size)
    x2 = min(w, x + crop_size)
    
    # Calculate width and height of the crop region
    width = x2 - x1
    height = y2 - y1
    
    return x1, y1, width, height

def crop_events_with_filter(events_batch, gaze_pt, crop_size, img_shape):
    # Use crop_image_around_point to get the crop region
    x1, y1, width, height = crop_image_around_point(img_shape, gaze_pt, crop_size)
    
    # Create the event filter with the specified region
    event_filter = dv.EventRegionFilter((x1, y1, width, height))
    
    # Apply the filter to the events batch
    event_filter.accept(events_batch)
    
    return event_filter.generateEvents()

