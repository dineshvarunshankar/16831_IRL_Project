import cv2

# Initialize the camera once
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame in a loop
    ret, frame = cap.read()
    
    # Check if frame was captured correctly
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('frame', frame)

    # Use waitKey(1) to allow the flow and check for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows when finished
cap.release()
cv2.destroyAllWindows()
