# SpaceWriting 
This Python code uses computer vision and machine learning to create an interactice space writing application using webcam. 
Hand Tracking: It uses the MediaPipe library to detect and track hand landmarks in real-time from the webcam feed.
1) Canvas Drawing:You can draw on a canvas using your index finger. Different colored buttons are provided for drawing. The canvas is displayed on the screen.
2) Erase and Detect Buttons: It has buttons on the canvas to erase the drawing or trigger digit detection.
Digit Recognition: When the detect button is pressed, the drawn digit on the canvas is processed. The region of interest is extracted, converted to grayscale, thresholded, and contours are found. The digit region is then resized, normalized, and fed into a pre-trained neural network (MNIST model) for digit recognition. The predicted digit is displayed on the canvas.
3) User Interface: It offers a simple user interface where you draw using hand gestures, erase, and trigger digit recognition.
4) Technologies Used:
OpenCV: For webcam access and image processing.
MediaPipe: For hand tracking.
TensorFlow: For loading and using a pre-trained MNIST digit recognition model.
5) Controls:Draw with the index finger.
Erase by clicking the erase button.
Trigger digit recognition by clicking the detect button.
6) Exit:
Press 'q' to exit the application.

This code combines real-time hand tracking, drawing, and digit recognition, providing an interactive and fun application that responds to hand gestures captured by the webcam.
