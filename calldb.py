import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import time

app = Flask(__name__)

db_path = r"C:\Users\Isuru\Downloads\finalone\junction_analysis.db"
output_dir = "dashboard_visualizations"
os.makedirs(output_dir, exist_ok=True)

def connect_to_db():
    print("Connecting to database...")
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        return conn, None
    except sqlite3.Error as e:
        return None, f"Error connecting to database: {str(e)}"

def get_tables(cursor):
    print("Fetching tables...")
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error:
        return []

def load_table_data(conn, table_name, max_rows=200, exclude_columns=None):
    print(f"Loading table {table_name}...")
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        
        if exclude_columns:
            columns = [col for col in columns if col not in exclude_columns]
        
        columns_str = ", ".join(columns)
        query = f"SELECT {columns_str} FROM {table_name} LIMIT {max_rows}"
        
        start_time = time.time()
        df_chunks = pd.read_sql_query(query, conn, chunksize=50)
        df = pd.concat(df_chunks, ignore_index=True)
        print(f"Loaded {table_name} in {time.time() - start_time:.2f} seconds")
        return df, None
    except (sqlite3.Error, pd.io.sql.DatabaseError, MemoryError) as e:
        return None, f"Error loading table {table_name}: {str(e)}"

def fetch_annotated_frame(conn, frame_id):
    print(f"Fetching frame {frame_id}...")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT annotated_frame FROM Detections WHERE frame_id = ? LIMIT 1", (frame_id,))
        result = cursor.fetchone()
        if result:
            blob_data = result[0]
            img = Image.open(BytesIO(blob_data))
            img_resized = img.resize((320, 240), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img_resized.save(buffered, format="PNG")
            return buffered.getvalue()
        return None
    except sqlite3.Error:
        return None

def blob_to_base64(blob_data):
    print("Converting BLOB to base64...")
    try:
        img = Image.open(BytesIO(blob_data))
        img_array = np.array(img)
        if img_array.shape[-1] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img_array)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception:
        return None

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    print("Starting dashboard route...")
    conn, conn_error = connect_to_db()
    if conn_error:
        print(f"Connection error: {conn_error}")
        return render_template('index.html', error=conn_error)

    cursor = conn.cursor()
    tables = get_tables(cursor)
    
    if not tables:
        print("No tables found in database.")
        conn.close()
        return render_template('index.html', error="No tables found in the database. Please rerun the video processing script to populate junction1_analysis.db.")

    frame_data_df = None
    detections_df = None
    annotations_df = None
    error_messages = []

    if "FrameData" in tables:
        frame_data_df, error = load_table_data(conn, "FrameData", max_rows=200)
        if error:
            error_messages.append(error)
            print(f"FrameData error: {error}")
        elif frame_data_df is not None and not frame_data_df.empty:
            frame_data_df['timestamp'] = pd.to_datetime(frame_data_df['timestamp'])
            print(f"FrameData loaded with {len(frame_data_df)} rows")

    if "Detections" in tables:
        detections_df, error = load_table_data(conn, "Detections", max_rows=200, exclude_columns=['annotated_frame'])
        if error:
            error_messages.append(error)
            print(f"Detections error: {error}")
        else:
            print(f"Detections loaded with {len(detections_df)} rows")

    if "Annotations" in tables:
        annotations_df, error = load_table_data(conn, "Annotations", max_rows=200)
        if error:
            error_messages.append(error)
            print(f"Annotations error: {error}")
        else:
            print(f"Annotations loaded with {len(annotations_df)} rows")

    overview = {}
    if frame_data_df is not None and not frame_data_df.empty:
        overview['total_frames'] = len(frame_data_df)
        vehicle_signals = frame_data_df['vehicle_signal'].value_counts()
        overview['green_signals'] = vehicle_signals.get("Green", 0)
        overview['red_signals'] = vehicle_signals.get("Red", 0)
    if detections_df is not None and not detections_df.empty:
        overview['total_detections'] = len(detections_df)
    if annotations_df is not None and not annotations_df.empty:
        overview['speed_annotations'] = len(annotations_df)

    counts_plot = None
    if frame_data_df is not None and not frame_data_df.empty:
        print("Generating counts plot...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame_data_df['timestamp'], y=frame_data_df['people_crossing'],
                                mode='lines', name='People Crossing', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=frame_data_df['timestamp'], y=frame_data_df['people_in_waiting_now'],
                                mode='lines', name='People Waiting', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=frame_data_df['timestamp'], y=frame_data_df['vehicles'],
                                mode='lines', name='Vehicles', line=dict(color='red')))
        fig.update_layout(title='Pedestrian and Vehicle Counts Over Time',
                         xaxis_title='Timestamp', yaxis_title='Count',
                         xaxis_tickangle=45, template='plotly_dark')
        counts_plot = fig.to_html(full_html=False)
        # Temporarily disable saving to file to avoid hang
        print("Skipping saving counts plot to file...")

    speeds_plot = None
    if frame_data_df is not None and not frame_data_df.empty:
        print("Generating speeds plot...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame_data_df['timestamp'], y=frame_data_df['avg_people_velocity_mps'],
                                mode='lines', name='Pedestrian Speed (m/s)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=frame_data_df['timestamp'], y=frame_data_df['avg_vehicle_speed_kmph'],
                                mode='lines', name='Vehicle Speed (km/h)', line=dict(color='red')))
        fig.update_layout(title='Average Pedestrian and Vehicle Speeds Over Time',
                         xaxis_title='Timestamp', yaxis_title='Speed',
                         xaxis_tickangle=45, template='plotly_dark')
        speeds_plot = fig.to_html(full_html=False)
        print("Skipping saving speeds plot to file...")

    vehicle_signal_plot = None
    pedestrian_signal_plot = None
    if frame_data_df is not None and not frame_data_df.empty:
        print("Generating vehicle signal plot...")
        vehicle_counts = frame_data_df['vehicle_signal'].value_counts().reset_index()
        vehicle_counts.columns = ['vehicle_signal', 'count']
        fig = px.bar(vehicle_counts, x='vehicle_signal', y='count', title='Vehicle Signal Distribution',
                     labels={'vehicle_signal': 'Signal', 'count': 'Count'}, color='vehicle_signal',
                     template='plotly_dark')
        vehicle_signal_plot = fig.to_html(full_html=False)
        print("Skipping saving vehicle signal plot to file...")

        print("Generating pedestrian signal plot...")
        pedestrian_counts = frame_data_df['pedestrian_signal'].value_counts().reset_index()
        pedestrian_counts.columns = ['pedestrian_signal', 'count']
        fig = px.bar(pedestrian_counts, x='pedestrian_signal', y='count', title='Pedestrian Signal Distribution',
                     labels={'pedestrian_signal': 'Signal', 'count': 'Count'}, color='pedestrian_signal',
                     template='plotly_dark')
        pedestrian_signal_plot = fig.to_html(full_html=False)
        print("Skipping saving pedestrian signal plot to file...")

    class_plot = None
    region_plot = None
    if detections_df is not None and not detections_df.empty:
        print("Generating class plot...")
        class_counts = detections_df['class_name'].value_counts().reset_index()
        class_counts.columns = ['class_name', 'count']
        fig = px.bar(class_counts, x='class_name', y='count', title='Detected Object Classes',
                     labels={'class_name': 'Class Name', 'count': 'Count'}, color='class_name',
                     template='plotly_dark')
        class_plot = fig.to_html(full_html=False)
        print("Skipping saving class plot to file...")

        print("Generating region plot...")
        region_counts = detections_df['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        fig = px.bar(region_counts, x='region', y='count', title='Detections by Region',
                     labels={'region': 'Region', 'count': 'Count'}, color='region',
                     template='plotly_dark')
        region_plot = fig.to_html(full_html=False)
        print("Skipping saving region plot to file...")

    speed_plot = None
    if annotations_df is not None and not annotations_df.empty:
        print("Generating speed plot...")
        fig = px.box(annotations_df, x='speed_unit', y='speed', color='speed_unit',
                     title='Speed Annotations by Unit', labels={'speed_unit': 'Speed Unit', 'speed': 'Speed'},
                     template='plotly_dark')
        speed_plot = fig.to_html(full_html=False)
        print("Skipping saving speed plot to file...")

    frame_image = None
    frame_id = request.form.get('frame_id', 1, type=int)
    if detections_df is not None and not detections_df.empty:
        max_frame_id = int(detections_df['frame_id'].max()) if not detections_df.empty else 1
        frame_id = min(max(1, frame_id), max_frame_id)
        start_time = time.time()
        blob_data = fetch_annotated_frame(conn, frame_id)
        print(f"Fetched frame {frame_id} in {time.time() - start_time:.2f} seconds")
        if blob_data:
            frame_image = blob_to_base64(blob_data)
            if frame_image:
                img_array = blob_to_image(blob_data)
                if img_array is not None:
                    frame_path = os.path.join(output_dir, f'frame_{frame_id}.png')
                    cv2.imwrite(frame_path, img_array)

    frame_data_html = frame_data_df.head(50).to_html(classes='table table-dark table-striped', index=False) if frame_data_df is not None and not frame_data_df.empty else None
    detections_html = detections_df.head(50).to_html(classes='table table-dark table-striped', index=False) if detections_df is not None and not detections_df.empty else None
    annotations_html = annotations_df.head(50).to_html(classes='table table-dark table-striped', index=False) if annotations_df is not None and not annotations_df.empty else None

    conn.close()
    print("Rendering template...")
    
    return render_template('index.html',
                           overview=overview,
                           counts_plot=counts_plot,
                           speeds_plot=speeds_plot,
                           vehicle_signal_plot=vehicle_signal_plot,
                           pedestrian_signal_plot=pedestrian_signal_plot,
                           class_plot=class_plot,
                           region_plot=region_plot,
                           speed_plot=speed_plot,
                           frame_image=frame_image,
                           frame_id=frame_id,
                           max_frame_id=max_frame_id if detections_df is not None else 1,
                           frame_data_html=frame_data_html,
                           detections_html=detections_html,
                           annotations_html=annotations_html,
                           errors=error_messages)

def blob_to_image(blob_data):
    print("Converting BLOB to image...")
    try:
        img = Image.open(BytesIO(blob_data))
        img_array = np.array(img)
        if img_array.shape[-1] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    except Exception:
        return None

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)  # Disable auto-reload