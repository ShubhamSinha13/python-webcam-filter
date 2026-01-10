#import libraries
import cv2
import numpy as np
#loading the model and importing then image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
sunglasses = cv2.imread(r"C:\Users\SHUBHAM\OneDrive\Desktop\New folder\sunglasses.png", cv2.IMREAD_UNCHANGED)

#video capturing and logic 
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Resize sunglasses to fit the face width
        sunglass_width = int(w)
        sunglass_height = int(sunglasses.shape[0] * (sunglass_width / sunglasses.shape[1]))
        resized_sunglasses = cv2.resize(sunglasses, (sunglass_width, sunglass_height))

        # Calculate position for sunglasses
        y1 = y + int(h*0.001) 
        y2 = y1 + sunglass_height
        x1 = x
        x2 = x + sunglass_width


        alpha = resized_sunglasses[:, :, 3] / 255.0 
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha * resized_sunglasses[:, :, c] +
                                      (1 - alpha) * frame[y1:y2, x1:x2, c])
    cv2.imshow('Sunglasses Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 






