# EXP-12--PROJECT-Face-Detection-with-Haar-Cascades
## Aim:

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required:

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm:

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:
```
import sys
!{sys.executable} -m pip install opencv-contrib-python --timeout 120 --retries 5
import numpy as np
import cv2 
import matplotlib.pyplot as plt
model = cv2.imread('VIJAY.jpg',0)
withglass = cv2.imread('83fe8b8d907f9b35f707d3a9488c77f1.jpg',0)
group = cv2.imread('EVERY.jpg',0)

plt.figure(figsize=(20,10))
plt.subplot(131);plt.imshow(cv2.resize(model, (1000, 1000)),cmap='gray');plt.title("Model")
plt.subplot(132);plt.imshow(cv2.resize(withglass, (1000, 1000)),cmap='gray');plt.title("Model with glass")
plt.subplot(133);plt.imshow(cv2.resize(group, (1000, 1000)),cmap='gray');plt.title("Group")
plt.show()

face_cascade_path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        f"Face cascade could not be loaded.\nPath: {face_cascade_path}"
    )

print("Face cascade loaded successfully!")
print(face_cascade_path)
def detect_face(img):
    face_img = img.copy()

    # Convert to grayscale if the image is colored
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    face_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in face_rects:
        cv2.rectangle(
            face_img,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

    return face_img
result = detect_face(withglass)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face Detection - Model with Glass")
plt.axis('off')
plt.show()
result = detect_face(model)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face Detection - Model")
plt.axis('off')
plt.show()
import cv2

print("OpenCV:", cv2.__version__)
print("Haar folder:", cv2.data.haarcascades)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\admin\haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    r"C:\Users\admin\haarcascade_eye.xml"
)

print("Face cascade empty:", face_cascade.empty())
print("Eye cascade empty:", eye_cascade.empty())
def detect_eyes(img):
    face_img = img.copy()

    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50)
    )

    print("Faces detected:", len(faces))

    for (x, y, w, h) in faces:

        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # Only upper part of face for eye detection
        roi_gray = gray[y:y+int(h*0.6), x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(15, 15)
        )

        print("Eyes detected:", len(eyes))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_img,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 255, 255),
                2
            )

    return face_img
result = detect_eyes(withglass)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face and Eye Detection")
plt.axis('off')
plt.show()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the camera. "
        "Check whether your webcam is connected or being used by another application."
    )

plt.ion()

fig, ax = plt.subplots(figsize=(10, 7))

ret, frame = cap.read()

if not ret:
    cap.release()
    plt.close(fig)
    raise RuntimeError("Could not read the first frame from the camera.")

frame = detect_face(frame)

im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
ax.set_title("Video Face Detection")
ax.axis('off')

while plt.fignum_exists(fig.number):

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from camera.")
        break

    frame = detect_face(frame)

    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    plt.pause(0.01)

cap.release()
plt.ioff()
plt.close(fig)


```

## Output:

<img width="1606" height="519" alt="download" src="https://github.com/user-attachments/assets/dc8f7621-49d4-41b6-9b49-a4128d32d473" />
<img width="366" height="656" alt="download" src="https://github.com/user-attachments/assets/08d07794-593a-4344-82b7-c2aa3a2021af" />
<img width="482" height="656" alt="download" src="https://github.com/user-attachments/assets/f498d006-f0f0-47cc-a4c4-a18eb85f76b4" />
<img width="482" height="656" alt="download" src="https://github.com/user-attachments/assets/e1f1b2c7-1b0e-4b87-a0f6-cc63c9c5db2e" />
<img width="738" height="579" alt="download" src="https://github.com/user-attachments/assets/de4b2f6e-07b7-45ab-86f1-2b4d0c787412" />





## Result:
Thus, the Python program for Face Detection using Haar Cascades with OpenCV and Matplotlib is implemented and executed successfully.
