
# TASK 3A

This is simple python code for detecting aruco marker from dicitonay ``6X6_250`` of ``cv2.aruco`` 

```
# import the required libraries

import cv2
import cv2.aruco as aruco

# Load the video
cap = cv2.VideoCapture('<path to file>\\aruco.mp4') #taking a pre recorded video as input

# Checking if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Defining the library which we will use . You can select whatever library you like, you just have to change "DICT_6X6_250" 
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

while cap.isOpened():
    ret, frame = cap.read()  # here ter is the boolean condition for success and frame is the actual frame by frame data from video 
    if not ret:
        break

    #  Converts the frame (color image) to grayscale for Aruco detection, which typically works better in grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''This is the core function for Aruco detection. It takes the grayscale image (gray), 
    the Aruco dictionary (aruco_dict), and detection parameters (parameters). 
    It returns three outputs:
    corners: A list of corner points for each detected marker (represented as lists of coordinates).
    ids: A list of IDs for each detected marker (corresponding to the IDs encoded in the markers).
    rejected: A list of rejected marker candidates that might not be valid Aruco markers.'''

    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers and bounding boxes
    if ids is not None:             #checking atleast one aruco marker is present or not 
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        for corner in corners:
            pts = corner.reshape((4, 2))
            pts = pts.astype(int)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw bounding box
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    else:
        print("No markers detected")        # if no aruco marker is present the it will print this

    # Display the frame
    cv2.imshow('Aruco Markers', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
