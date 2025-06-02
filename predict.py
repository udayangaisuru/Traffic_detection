import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import sqlite3
import os
import io
from PIL import Image
import serial
import time

# Initialize YOLO model
model_yolo = YOLO("yolov8n.pt")

# Load trained LSTM model
model_lstm = load_model('lstm_traffic_model.h5')

# Video file path
video_path = r"C:\Users\Isuru\Downloads\finalone\Input Video\test1.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Image size: {frame_width}x{frame_height}")

# Define ROIs
crosswalk_roi = np.array([[0, 600], [1500, 700], [1500, 1000], [0, 1000]], dtype=np.int32)
right_side_roi = np.array([
    [int(0.75 * frame_width), frame_height - 100],
    [frame_width, frame_height - 100],
    [frame_width, frame_height],
    [int(0.75 * frame_width), frame_height]
], dtype=np.int32)
print(f"Crosswalk ROI: {crosswalk_roi}")
print(f"Right-side ROI: {right_side_roi}")

# Initialize SQLite database
db_path = "junction1_analysis.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop existing tables to prevent duplicate frame_id errors
cursor.execute('DROP TABLE IF EXISTS FrameData')
cursor.execute('DROP TABLE IF EXISTS Detections')
cursor.execute('DROP TABLE IF EXISTS Annotations')

# Create tables
cursor.execute('''
    CREATE TABLE FrameData (
        frame_id INTEGER PRIMARY KEY,
        timestamp TEXT,
        people_crossing INTEGER,
        people_in_crosswalk_now INTEGER,
        avg_people_velocity_mps REAL,
        people_waiting INTEGER,
        people_in_waiting_now INTEGER,
        avg_people_velocity_waiting_mps REAL,
        vehicles INTEGER,
        avg_vehicle_speed_kmph REAL,
        queued_vehicles INTEGER,
        vehicle_signal TEXT,
        pedestrian_signal TEXT
    )
''')
cursor.execute('''
    CREATE TABLE Detections (
        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id INTEGER,
        track_id INTEGER,
        class_name TEXT,
        bbox_x REAL,
        bbox_y REAL,
        bbox_w REAL,
        bbox_h REAL,
        region TEXT,
        annotated_frame BLOB,
        FOREIGN KEY (frame_id) REFERENCES FrameData(frame_id)
    )
''')
cursor.execute('''
    CREATE TABLE Annotations (
        annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id INTEGER,
        track_id INTEGER,
        speed REAL,
        speed_unit TEXT,
        FOREIGN KEY (frame_id) REFERENCES FrameData(frame_id)
    )
''')
conn.commit()

# Initialize serial communication with Arduino
try:
    ser = serial.Serial('COM5', 9600, timeout=1)  # Replace 'COM3' with your Arduino's port
    time.sleep(2)  # Wait for serial connection to initialize
    print("Serial connection established with Arduino")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

# Function to send command to Arduino
def send_to_arduino(command):
    if ser:
        try:
            ser.write((command + '\n').encode())
            print(f"Sent to Arduino: {command}")
            time.sleep(0.1)  # Small delay to ensure Arduino processes the command
        except serial.SerialException as e:
            print(f"Error sending to Arduino: {e}")

# Function to convert frame to BLOB
def frame_to_blob(frame_data):
    frame_resized = cv2.resize(frame_data, (640, 480))  # Resize to reduce size
    _, img_encoded = cv2.imencode('.png', frame_resized)
    return img_encoded.tobytes()

# Function to check if a point is inside the ROI
def is_point_in_roi(x, y, roi):
    return cv2.pointPolygonTest(roi, (x, y), False) >= 0

# Function to check if bounding box intersects ROI
def is_bbox_in_roi(x, y, w, h, roi):
    bottom_left = (x - w/2, y + h/2)
    bottom_right = (x + w/2, y + h/2)
    return is_point_in_roi(bottom_left[0], bottom_left[1], roi) >= 0 or \
           is_point_in_roi(bottom_right[0], bottom_right[1], roi) >= 0

# Initialize data storage and scaler
features = [
    'People_Crossing', 'People_In_Crosswalk_Now', 'Avg_People_Velocity_Crossing',
    'People_Waiting', 'People_In_Waiting_Now', 'Avg_People_Velocity_Waiting',
    'Vehicles', 'Avg_Vehicle_Speed', 'Queued_Vehicles'
]
data = []  # For scaler initialization
scaler = StandardScaler()
sequence_length = 8
sequence_buffer = []

# Signal state variables
current_vehicle_signal = "Green"
last_model_prediction = "Green"
previous_model_prediction = "Green"
vehicle_signal_display = "Green"  # Ensure this matches the initial state
pedestrian_signal_display = "Pedestrian Red"

# Timing variables for traffic light phases
last_signal_change = time.time()
min_phase_duration = 30  # Minimum duration for each phase (seconds)

# Tracking variables
prev_positions = {}
current_crossing_people = set()
prev_crossing_people = set()
current_waiting_people = set()
prev_waiting_people = set()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    # Run YOLO detection and tracking
    results = model_yolo.track(frame, persist=True, classes=[0, 2, 3, 5, 7])

    # Initialize frame metrics
    people_in_crosswalk_now = 0
    people_velocities_crossing = []
    people_in_waiting_now = 0
    people_velocities_waiting = []
    num_vehicles = 0
    vehicle_speeds = []
    queued_vehicles = 0
    motorbike_in_crosswalk = False

    current_crossing_people.clear()
    current_waiting_people.clear()

    # Process detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, classes):
            x, y, w, h = box
            pixel_to_meter = 0.05  # 1 pixel = 5 cm

            speed = None
            speed_unit = None
            region = None

            if cls == 0:  # Person
                if is_bbox_in_roi(x, y, w, h, crosswalk_roi):
                    current_crossing_people.add(track_id)
                    people_in_crosswalk_now += 1
                    region = "Crosswalk"
                    if track_id in prev_positions:
                        prev_x, prev_y, prev_frame = prev_positions[track_id]
                        time_diff = (frame_count - prev_frame) / fps
                        if time_diff > 0:
                            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                            speed_mps = distance * pixel_to_meter / time_diff  # m/s
                            people_velocities_crossing.append(speed_mps)
                            speed = speed_mps
                            speed_unit = "m/s"

                if is_bbox_in_roi(x, y, w, h, right_side_roi):
                    current_waiting_people.add(track_id)
                    people_in_waiting_now += 1
                    region = region or "Waiting Area"
                    if track_id in prev_positions:
                        prev_x, prev_y, prev_frame = prev_positions[track_id]
                        time_diff = (frame_count - prev_frame) / fps
                        if time_diff > 0:
                            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                            speed_mps = distance * pixel_to_meter / time_diff  # m/s
                            people_velocities_waiting.append(speed_mps)
                            speed = speed or speed_mps
                            speed_unit = speed_unit or "m/s"

            elif cls == 3:  # Motorbike
                if is_bbox_in_roi(x, y, w, h, crosswalk_roi):
                    motorbike_in_crosswalk = True
                num_vehicles += 1
                region = "Road"
                if track_id in prev_positions:
                    prev_x, prev_y, prev_frame = prev_positions[track_id]
                    time_diff = (frame_count - prev_frame) / fps
                    if time_diff > 0:
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        speed_mps = distance * pixel_to_meter / time_diff
                        speed_kmph = speed_mps * 3.6  # km/h
                        vehicle_speeds.append(speed_kmph)
                        speed = speed_kmph
                        speed_unit = "km/h"
                        if speed_kmph < 5:
                            queued_vehicles += 1

            elif cls in [2, 5, 7]:  # Other vehicle classes
                num_vehicles += 1
                region = "Road"
                if track_id in prev_positions:
                    prev_x, prev_y, prev_frame = prev_positions[track_id]
                    time_diff = (frame_count - prev_frame) / fps
                    if time_diff > 0:
                        distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        speed_mps = distance * pixel_to_meter / time_diff
                        speed_kmph = speed_mps * 3.6  # km/h
                        vehicle_speeds.append(speed_kmph)
                        speed = speed_kmph
                        speed_unit = "km/h"
                        if speed_kmph < 5:
                            queued_vehicles += 1

            prev_positions[track_id] = (x, y, frame_count)

            # Save detection details
            if region:
                class_name = {0: "Person", 2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck"}.get(cls, "Unknown")
                annotated_frame_blob = frame_to_blob(results[0].plot())
                cursor.execute('''
                    INSERT INTO Detections (frame_id, track_id, class_name, bbox_x, bbox_y, bbox_w, bbox_h, region, annotated_frame)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (frame_count, track_id, class_name, x, y, w, h, region, annotated_frame_blob))

                # Save speed annotation
                if speed is not None:
                    cursor.execute('''
                        INSERT INTO Annotations (frame_id, track_id, speed, speed_unit)
                        VALUES (?, ?, ?, ?)
                    ''', (frame_count, track_id, speed, speed_unit))

    conn.commit()

    prev_crossing_people = current_crossing_people.copy()
    prev_waiting_people = current_waiting_people.copy()

    # Adjust people counts if motorbike is in crosswalk
    if motorbike_in_crosswalk:
        people_in_crosswalk_now = max(0, people_in_crosswalk_now - 1)
        current_crossing_people_count = max(0, len(current_crossing_people) - 1)
    else:
        current_crossing_people_count = len(current_crossing_people)

    avg_people_velocity_crossing = np.mean(people_velocities_crossing) if people_velocities_crossing else 0
    avg_people_velocity_waiting = np.mean(people_velocities_waiting) if people_velocities_waiting else 0
    avg_vehicle_speed = np.mean(vehicle_speeds) if vehicle_speeds else 0

    # Collect frame data
    frame_data = {
        "Timestamp": timestamp,
        "People_Crossing": current_crossing_people_count,
        "People_In_Crosswalk_Now": people_in_crosswalk_now,
        "Avg_People_Velocity_Crossing": avg_people_velocity_crossing,  # m/s
        "People_Waiting": people_in_waiting_now,
        "People_In_Waiting_Now": people_in_waiting_now,
        "Avg_People_Velocity_Waiting": avg_people_velocity_waiting,  # m/s
        "Vehicles": num_vehicles,
        "Avg_Vehicle_Speed": avg_vehicle_speed,  # km/h
        "Queued_Vehicles": queued_vehicles
    }
    data.append(frame_data)  # For scaler initialization

    # Save frame data to database
    cursor.execute('''
        INSERT INTO FrameData (
            frame_id, timestamp, people_crossing, people_in_crosswalk_now, avg_people_velocity_mps,
            people_waiting, people_in_waiting_now, avg_people_velocity_waiting_mps,
            vehicles, avg_vehicle_speed_kmph, queued_vehicles, vehicle_signal, pedestrian_signal
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        frame_count, timestamp, frame_data["People_Crossing"], frame_data["People_In_Crosswalk_Now"],
        frame_data["Avg_People_Velocity_Crossing"], frame_data["People_Waiting"],
        frame_data["People_In_Waiting_Now"], frame_data["Avg_People_Velocity_Waiting"],
        frame_data["Vehicles"], frame_data["Avg_Vehicle_Speed"], frame_data["Queued_Vehicles"],
        vehicle_signal_display, pedestrian_signal_display  # Use display values for consistency
    ))
    conn.commit()

    # Prepare features for prediction
    X_frame = np.array([[frame_data[feat] for feat in features]])
    if len(data) == 1:
        scaler.fit(X_frame)
    X_scaled = scaler.transform(X_frame)

    # Apply feature weights
    weights = np.ones(len(features))
    weights[0] = 1.0  # People_Crossing
    weights[1] = 1.0  # People_In_Crosswalk_Now
    weights[2] = 1.0  # Avg_People_Velocity_Crossing
    weights[3] = 1.0  # People_Waiting
    weights[4] = 8.0  # People_In_Waiting_Now
    weights[5] = 1.0  # Avg_People_Velocity_Waiting
    weights[6] = 1.0  # Vehicles
    weights[7] = 1.0  # Avg_Vehicle_Speed
    weights[8] = 1.0  # Queued_Vehicles
    X_weighted = X_scaled * weights

    sequence_buffer.append(X_weighted[0])
    if len(sequence_buffer) > sequence_length:
        sequence_buffer.pop(0)

    # Get model prediction if enough frames
    if len(sequence_buffer) == sequence_length:
        X_seq = np.array([sequence_buffer])
        pred = model_lstm.predict(X_seq, verbose=0)
        pred_label = int(pred[0] > 0.5)
        last_model_prediction = "Green" if pred_label == 1 else "Red"

    # Manage vehicle signal state with timing
    current_time = time.time()
    if (current_time - last_signal_change) >= min_phase_duration or last_model_prediction != previous_model_prediction:
        if last_model_prediction == "Red" and previous_model_prediction == "Green" and len(sequence_buffer) == sequence_length:
            current_vehicle_signal = "Yellow"
            vehicle_signal_display = "Yellow"
            send_to_arduino("V_Y,10")  # Yellow for 10 seconds
        elif current_vehicle_signal == "Yellow":
            current_vehicle_signal = "Red"
            vehicle_signal_display = "Red"
            send_to_arduino("P_G,30")  # Pedestrian green for 30 seconds
        else:
            current_vehicle_signal = last_model_prediction
            vehicle_signal_display = last_model_prediction
            if current_vehicle_signal == "Green":
                send_to_arduino("V_G,40")  # Vehicle green for 40 seconds
            else:
                send_to_arduino("P_G,30")  # Pedestrian green for 30 seconds
        
        last_signal_change = current_time
        previous_model_prediction = last_model_prediction

    # Determine pedestrian signal
    if vehicle_signal_display in ["Green", "Yellow"]:  # Use display value for consistency
        pedestrian_signal = "Red"
        pedestrian_signal_display = "Pedestrian Red"
    else:
        pedestrian_signal = "Green"
        pedestrian_signal_display = "Pedestrian Green"

    # Draw ROIs and annotations
    annotated_frame = results[0].plot()
    cv2.polylines(annotated_frame, [crosswalk_roi], True, (0, 255, 0), 2)
    cv2.polylines(annotated_frame, [right_side_roi], True, (0, 0, 255), 2)

    # Overlay vehicle signal
    vehicle_color = (0, 255, 0) if vehicle_signal_display == "Green" else (0, 165, 255) if vehicle_signal_display == "Yellow" else (0, 0, 255)
    cv2.putText(annotated_frame, f"Vehicle: {vehicle_signal_display}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, vehicle_color, 2, cv2.LINE_AA)

    # Overlay pedestrian signal
    pedestrian_color = (0, 255, 0) if pedestrian_signal_display == "Pedestrian Green" else (0, 0, 255)
    cv2.putText(annotated_frame, pedestrian_signal_display, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, pedestrian_color, 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow("YOLO Tracking with Predictions", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()
if ser:
    ser.close()
print(f"Data saved to {db_path}")