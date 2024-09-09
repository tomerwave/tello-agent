from djitellopy import Tello
import cv2
import numpy as np
import mediapipe as mp

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Create a Tello object
tello = Tello()

# Connect to the Tello drone
tello.connect()

# Start the video stream
tello.streamon()

# OpenCV window to display the video feed
cv2.namedWindow("Tello Video Feed")

try:
    while True:
        # Get the video frame
        frame = tello.get_frame_read().frame

        # Resize the frame for faster processing (optional)
        frame = cv2.resize(frame, (640, 480))

        # Prepare the frame for YOLO object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Analyzing the outputs from YOLO
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold for detecting a person
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression to remove duplicate detections
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Check for person detection and then hand gesture
        for i in range(len(boxes)):
            if i in indexes and classes[class_ids[i]] == "person":
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Hand detection using MediaPipe within the detected person's area
                person_frame = frame[y:y+h, x:x+w]
                person_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                result = hands.process(person_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(person_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Extract landmark positions to recognize gestures
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                        # Example gesture detection logic: Wave (hand raised with open palm)
                        if wrist.y > index_tip.y and wrist.y > thumb_tip.y:
                            cv2.putText(frame, "Wave Detected", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detections
        cv2.imshow("Tello Video Feed", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Safely close the connection to the drone and video stream
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
