import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os
import csv
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, scrolledtext
import tensorflow as tf

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Placeholder function definitions
def calculate_angle(point1, point2, point3):
    """Calculate the angle between three points."""
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_angle(image, point1, point2, point3, angle, color):
    """Draw the angle on the image."""
    cv2.putText(image, str(int(angle)), 
                tuple(np.multiply(point2, [1, 1]).astype(int)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# Setup variables for calibration
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angles = []
calibration_neck_angles = []

shoulder_threshold = 0
neck_threshold = 0

last_alert_time = 0
alert_cooldown = 10  # seconds
sound_file = "alert.wav"  # path to your alert sound file

# Setup logging
log_file = 'posture_log.csv'
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Shoulder Angle", "Neck Angle", "Status"])

# GUI for settings
root = tk.Tk()
root.withdraw()
custom_alert_cooldown = simpledialog.askinteger("Settings", "Set alert cooldown (seconds):", initialvalue=10)
custom_shoulder_threshold = simpledialog.askinteger("Settings", "Set shoulder threshold:", initialvalue=10)
custom_neck_threshold = simpledialog.askinteger("Settings", "Set neck threshold:", initialvalue=10)

# Function to view posture logs
def view_logs():
    log_window = tk.Toplevel(root)
    log_window.title("Posture Logs")
    log_window.geometry("600x400")
    st = scrolledtext.ScrolledText(log_window, width=70, height=20)
    st.pack(pady=10)
    with open(log_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            st.insert(tk.END, ','.join(row) + '\n')
    st.configure(state='disabled')

# Add a menu to the root window
root.deiconify()
menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="View Logs", command=view_logs)
file_menu.add_command(label="Exit", command=root.quit)
root.withdraw()

# Load the trained ML model
model = tf.keras.models.load_model('posture_model.h5')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Pose Detection
        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))

        # Angle Calculation
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        # Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - custom_shoulder_threshold
            neck_threshold = np.mean(calibration_neck_angles) - custom_neck_threshold
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))

        # Feedback
        if is_calibrated:
            img = cv2.resize(frame, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            posture = 'Good Posture' if prediction[0][0] > 0.5 else 'Poor Posture'
            
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > custom_alert_cooldown:
                    if os.path.exists(sound_file):
                        playsound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            # Log data
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), shoulder_angle, neck_angle, status])

            cv2.putText(frame, posture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
