import cv2
import imutils
import mediapipe as mp

# Initialize MediaPipe objects
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize variables
pushup_count = 0
shoulder_landmarks = [11, 12]  # Landmarks for shoulders
elbow_landmarks = [13, 14]     # Landmarks for elbows
is_down = False

# Initialize video capture (use the default camera, change if needed)
cap = cv2.VideoCapture(0)

# Create a pose estimation object
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = imutils.resize(frame, width=800)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Get landmark positions
        if results.pose_landmarks:
            shoulder_left = results.pose_landmarks.landmark[shoulder_landmarks[0]]
            shoulder_right = results.pose_landmarks.landmark[shoulder_landmarks[1]]
            elbow_left = results.pose_landmarks.landmark[elbow_landmarks[0]]
            elbow_right = results.pose_landmarks.landmark[elbow_landmarks[1]]

            # Check if arms are extended (push-up position)
            if shoulder_left.y < elbow_left.y and shoulder_right.y < elbow_right.y:
                if not is_down:
                    # Arm position changed from up to down
                    is_down = True
            else:
                if is_down:
                    # Arm position changed from down to up (completed a push-up)
                    pushup_count += 1
                    is_down = False

        # Display push-up count
        cv2.putText(frame, f"Push-ups: {pushup_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Push-up Counter", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
