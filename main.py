import os
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tensorflow as tf

# Suppress TensorFlow warnings and information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off OneDNN custom operations

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Adjust confidence levels as needed

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Function to handle button click events
def handle_button_click(event, x, y, flags, param):
    global current_color, canvas, finger_points, detect_button_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the erase button is clicked
        if erase_button[0] < x < erase_button[2] and erase_button[1] < y < erase_button[3]:
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Reset the canvas
            finger_points.clear()  # Clear the finger trajectory points
            detect_button_pressed = False  # Reset the detect button state

        # Check if the detect button is clicked
        elif detect_button[0] < x < detect_button[2] and detect_button[1] < y < detect_button[3]:
            detect_button_pressed = not detect_button_pressed  # Toggle the detect button state

        # Check if a color button is clicked
        for i, button in enumerate(color_buttons):
            if button[0] < x < button[2] and button[1] < y < button[3]:
                update_current_color(i)

def update_current_color(button_index):
    global current_color

    if button_index == 0:
        current_color = (255, 0, 0)  # Red
    elif button_index == 1:
        current_color = (0, 255, 0)  # Green
    elif button_index == 2:
        current_color = (0, 0, 255)  # Blue

# Open webcam
cap = cv2.VideoCapture(0)

# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Button positions
erase_button = (10, 10, 110, 60)  # Rectangle coordinates (x1, y1, x2, y2)
detect_button = (130, 10, 230, 60)  # Detect button coordinates
color_buttons = [(250, 10, 350, 60), (370, 10, 470, 60)]  # Two color buttons

# Create a canvas
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Initialize finger trajectory points
finger_points = deque(maxlen=512)

# Color variables
current_color = (0, 255, 0)  # Initial color (green)

# Detect button state
detect_button_pressed = False

# Set the callback function for mouse events
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', handle_button_click)

# Load MNIST dataset using tensorflow.keras.datasets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on the MNIST data
model.fit(train_images, train_labels, epochs=5)

def draw_buttons():
    cv2.rectangle(canvas, erase_button[:2], erase_button[2:], (255, 255, 255), -1)  # Draw erase button with white background
    cv2.putText(canvas, 'Erase', (erase_button[0] + 10, erase_button[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # Add text to erase button

    cv2.rectangle(canvas, detect_button[:2], detect_button[2:], (255, 255, 255), -1)  # Draw detect button with white background
    cv2.putText(canvas, 'Detect', (detect_button[0] + 10, detect_button[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # Add text to detect button

    for i, button in enumerate(color_buttons):
        button_color = (255, 0, 0) if i == 0 else (0, 255, 0) if i == 1 else (0, 0, 255)
        cv2.rectangle(canvas, button[:2], button[2:], button_color, -1)

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip
            index_finger_tip = tuple(
                np.multiply(
                    [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                     hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y],
                    [frame_width, frame_height]
                ).astype(int)
            )

            # Append the index finger tip to the finger trajectory points
            finger_points.appendleft(index_finger_tip)

            # Draw lines on the canvas connecting the finger trajectory points
            for i in range(1, len(finger_points)):
                if finger_points[i - 1] is not None and finger_points[i] is not None:
                    cv2.line(canvas, finger_points[i - 1], finger_points[i], current_color, 2)

    # Draw buttons on the canvas
    draw_buttons()

    # Detect digits when the detect button is pressed
    if detect_button_pressed:
        # Get the region of interest (ROI) containing the drawn image
        roi = canvas[10:410, 10:410]  # Assuming canvas size is 400x400 starting from (10, 10)

        # Convert the image to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary image
        _, roi_threshold = cv2.threshold(roi_gray, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the maximum area (assuming it's the digit)
        max_contour = max(contours, key=cv2.contourArea, default=None)

        if max_contour is not None:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(max_contour)

            # Extract the digit region from the thresholded image
            digit_roi = roi_threshold[y:y+h, x:x+w]

            # Resize the digit region to match the MNIST input size (28x28)
            digit_resized = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalize pixel values to be between 0 and 1
            digit_normalized = digit_resized / 255.0

            # Reshape the image to match the input shape expected by the model
            input_image = np.reshape(digit_normalized, (1, 28, 28))

            # Make a prediction using the trained model
            predictions = model.predict(input_image)

            # Get the predicted digit
            predicted_digit = np.argmax(predictions)

            # Display the predicted digit on the canvas
            cv2.putText(canvas, str(predicted_digit), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # Reset detect button state
        detect_button_pressed = False

    # Show the frame with hand landmarks
    cv2.imshow('Hand Tracking', frame)

    # Show the canvas with finger movement tracking
    cv2.imshow('Canvas', canvas)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
