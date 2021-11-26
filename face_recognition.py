import cv2
import dlib

#read the image
img = cv2.imread("anh1.png")


# convert img to grayscale: 3D -> 2D
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# dlib: load the face recognition detector
face_detector = dlib.get_frontal_face_detector()

# load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# use detector to find face landmarks
faces = face_detector(gray)  #faces variable is a array with number of face in the picture

for face in faces:
    x1 = face.left() #left point
    y1 = face.top() #top point
    x2 = face.right() #right point
    y2 = face.bottom() #bottom point

    #draw rectangle
    cv2.rectangle(img= img, pt1=(x1, y1), pt2=(x2, y2), 
                 color=(0,255,0), thickness=3)
    face_feature = predictor(image=gray, box=face)

    # loop through all 68 points
    for n in range(0, 68):
        x = face_feature.part(n).x
        y = face_feature.part(n).y

        # draw a circle
        cv2.circle(img=img,center=(x,y),radius=2,
                   color=(255,0,0), thickness=1)

# show the img
imgSize = cv2.resize(img, (680,680))
cv2.imshow(winname="Face Recognition App",mat=imgSize)

# wait for a key press to exit
cv2.waitKey(delay=0)

# close img
cv2.destroyAllWindows()