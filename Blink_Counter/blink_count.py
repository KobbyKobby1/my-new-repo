import cv2
import dlib
from scipy.spatial import distance as dist

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_points, facial_landmarks):
    # Compute the Euclidean distances between the vertical eye landmarks
    P2_P6 = dist.euclidean((facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y))
    P3_P5 = dist.euclidean((facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y))

    # Compute the Euclidean distance between the horizontal eye landmarks
    P1_P4 = dist.euclidean((facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y))

    # Compute the EAR
    ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    return ear


# Load the pre-trained facial landmark predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize blink counter and frame counter
blink_count = 0
frame_counter = 0
blink_threshold = 0.2  # Adjust this threshold as needed
consecutive_frames = 3  # Number of consecutive frames the eye must be below the threshold

# Open the webcam
cap = cv2.VideoCapture(0)

# Indices of the left and right eye landmarks
left_eye_indices = [36, 37, 38, 39, 40, 41]
right_eye_indices = [42, 43, 44, 45, 46, 47]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)
    
    # Loop over the face detections
    for face in faces:
        landmarks = predictor(gray, face)

        # Calculate EAR for both eyes
        left_eye_ear = calculate_ear(left_eye_indices, landmarks)
        right_eye_ear = calculate_ear(right_eye_indices, landmarks)

        # Check if either eye is below the blink threshold
        if left_eye_ear < blink_threshold or right_eye_ear < blink_threshold:
            frame_counter += 1
        else:
            # If the eyes were closed for a sufficient number of frames, increment the blink count
            if frame_counter >= consecutive_frames:
                blink_count += 1
            # Reset the frame counter
            frame_counter = 0

    # Display the frame with blink count
    cv2.putText(frame, f'Blinks: {blink_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Blink Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
