import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Video file path
video_path = r"C:\Users\Isuru\Downloads\finalone\Input Video\test2.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame size: {frame_width}x{frame_height}")

# Define crosswalk ROI
crosswalk_roi = np.array([[0, 700], [1500, 700], [1500, 1000], [0, 1000]], dtype=np.int32)

# Define right-side ROI (e.g., right 1/4 of frame, height 100 pixels from bottom)
right_side_roi = np.array([
    [int(0.75 * frame_width), frame_height - 100],  # Top-left
    [frame_width, frame_height - 100],            # Top-right
    [frame_width, frame_height],                  # Bottom-right
    [int(0.75 * frame_width), frame_height]       # Bottom-left
], dtype=np.int32)

print(f"Crosswalk ROI: {crosswalk_roi}")
print(f"Right-side ROI: {right_side_roi}")

# Function to check if a point is inside the ROI
def is_point_in_roi(x, y, roi):
    return cv2.pointPolygonTest(roi, (x, y), False) >= 0

# Function to check if bounding box intersects ROI (using bottom edge)
def is_bbox_in_roi(x, y, w, h, roi):
    bottom_left = (x - w/2, y + h/2)
    bottom_right = (x + w/2, y + h/2)
    return is_point_in_roi(bottom_left[0], bottom_left[1], roi) >= 0 or \
           is_point_in_roi(bottom_right[0], bottom_right[1], roi) >= 0

# CSV data storage
data = []

# Tracking variables
prev_positions = {}  # Store previous positions {track_id: (x, y, frame_num)}
current_crossing_people = set()    # People in crosswalk in current frame
prev_crossing_people = set()       # People in crosswalk in previous frame
current_waiting_people = set()     # People in right-side ROI in current frame
prev_waiting_people = set()        # People in right-side ROI in previous frame
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # Run YOLO detection and tracking
    results = model.track(frame, persist=True, classes=[0, 2, 3, 5, 7])

    # Initialize counts and velocities for this frame
    people_in_crosswalk_now = 0      # People in crosswalk in this frame
    people_velocities_crossing = []  # Velocities of people in crosswalk
    people_in_waiting_now = 0        # People in right-side ROI in this frame
    people_velocities_waiting = []   # Velocities of people in waiting area
    num_vehicles = 0
    vehicle_velocities = []
    queued_vehicles = 0

    # Clear current people sets for this frame
    current_crossing_people.clear()
    current_waiting_people.clear()

    # Process detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            x, y, w, h = box
            center = (x, y)
            pixel_to_meter = 0.1

            if cls == 0:  # Person
                # Check if person's bottom edge (feet) is in crosswalk ROI
                if is_bbox_in_roi(x, y, w, h, crosswalk_roi):
                    current_crossing_people.add(track_id)
                    people_in_crosswalk_now += 1

                    # Velocity estimation for people in the crosswalk
                    if track_id in prev_positions:
                        prev_x, prev_y, prev_frame = prev_positions[track_id]
                        time_diff = (frame_count - prev_frame) / fps
                        if time_diff > 0:
                            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                            speed_mps = distance * pixel_to_meter / time_diff
                            speed_kmph = speed_mps * 3.6
                            people_velocities_crossing.append(speed_kmph)

                # Check if person's bottom edge (feet) is in right-side waiting ROI
                if is_bbox_in_roi(x, y, w, h, right_side_roi):
                    current_waiting_people.add(track_id)
                    people_in_waiting_now += 1

                    # Velocity estimation for people waiting
                    if track_id in prev_positions:
                        prev_x, prev_y, prev_frame = prev_positions[track_id]
                        time_diff = (frame_count - prev_frame) / fps
                        if time_diff > 0:
                            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                            speed_mps = distance * pixel_to_meter / time_diff
                            speed_kmph = speed_mps * 3.6
                            people_velocities_waiting.append(speed_kmph)

            elif cls in [2, 3, 5, 7]:  # Vehicle classes
                num_vehicles += 1
                if track_id in prev_positions:
                    prev_x, prev_y, prev_frame = prev_positions[track_id]
                    time_diff = (frame_count - prev_frame) / fps
                    if time_diff > 0:
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        speed_mps = distance * pixel_to_meter / time_diff
                        speed_kmph = speed_mps * 3.6
                        vehicle_velocities.append(speed_kmph)
                        if speed_kmph < 5:
                            queued_vehicles += 1

            prev_positions[track_id] = (x, y, frame_count)

    # Update previous people sets for the next frame
    prev_crossing_people = current_crossing_people.copy()
    prev_waiting_people = current_waiting_people.copy()

    avg_people_velocity_crossing = np.mean(people_velocities_crossing) if people_velocities_crossing else 0
    avg_people_velocity_waiting = np.mean(people_velocities_waiting) if people_velocities_waiting else 0
    avg_vehicle_velocity = np.mean(vehicle_velocities) if vehicle_velocities else 0

    data.append({
        "Timestamp": timestamp,
        "People_Crossing": len(current_crossing_people),  # Live count of people in crosswalk
        "People_In_Crosswalk_Now": people_in_crosswalk_now,
        "Avg_People_Velocity_Crossing_kmph": avg_people_velocity_crossing,
        "People_Waiting": people_in_waiting_now,  # Live count of people in waiting area
        "People_In_Waiting_Now": people_in_waiting_now,  # Redundant, can be removed
        "Avg_People_Velocity_Waiting_kmph": avg_people_velocity_waiting,
        "Vehicles": num_vehicles,
        "Avg_Vehicle_Velocity_kmph": avg_vehicle_velocity,
        "Queued_Vehicles": queued_vehicles
    })

    # Draw ROIs on the frame
    frame_with_roi = frame.copy()
    cv2.polylines(frame_with_roi, [crosswalk_roi], True, (0, 255, 0), 2)  # Green for crosswalk
    cv2.polylines(frame_with_roi, [right_side_roi], True, (0, 0, 255), 2)  # Red for waiting area (changed to red for visibility)
    annotated_frame = results[0].plot()
    cv2.polylines(annotated_frame, [crosswalk_roi], True, (0, 255, 0), 2)
    cv2.polylines(annotated_frame, [right_side_roi], True, (0, 0, 255), 2)

    # Ensure the window stays open and displays
    cv2.imshow("YOLO Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv("junction_analysis_test2.csv", index=False)
print("Data saved to junction_analysis.csv")