import cv2
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# read the video
# Create a VideoCapture object and read from input file
# If the input is taken from the camera, pass 0 instead of the video file name.
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point
        # Draw a rectangle
        cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(
            x2, y2), color=(0, 255, 0), thickness=4)
        face_feature = predictor(image=frame,box=face)

        for n in range(0, 68):
            x = face_feature.part(n).x
            y = face_feature.part(n).y

            cv2.circle(img=frame, center=(x, y), radius=2,
                        color=(0, 255, 0), thickness=1)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed, Press Q on keyboard to  exit
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()