import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_face68_txt(face_img): 
    image = cv2.imread(face_img)
    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
	# Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        #for i, (x, y) in enumerate(shape):
	    # Draw the circle to mark the keypoint 
        #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		
    # Display the image
    #cv2.imshow('Landmark Detection', image)
    
    # Displaying the array
    print('face landmarks:\n', shape_np)
    fn=str(face_img)
    lms_fn=fn[:fn.rindex('.')]+'.txt'
    file = open(lms_fn, "w+")
 
    # Saving  face landmarks  in a text file
    # 1-68  x  y format
    for i in range(len(shape_np)):
        file.write('{0} {1} {2}\n'.format(i+1, shape_np[i][0],shape_np[i][1]))     
    file.close()

get_face68_txt("source1.jpg")
