import cv2
import cv2.aruco as aruco

# Load the video
cap = cv2.VideoCapture('aruco.mp4')

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the dictionary and parameters for Aruco detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers and bounding boxes
    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        for corner in corners:
            pts = corner.reshape((4, 2))
            pts = pts.astype(int)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw bounding box
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    else:
        print("No markers detected")

    # Display the frame
    cv2.imshow('Aruco Markers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()