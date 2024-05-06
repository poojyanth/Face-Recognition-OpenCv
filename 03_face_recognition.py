import cv2
import numpy as np
import os 
import tkinter as tk
import sys

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')


font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None', 'Poojyanth', 'Hemanth', 'Z', 'W'] 

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

blink_counter = 0
blink_threshold = 1
blink_detected = False


minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
Liveliness = False

def show_alert(str):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(str)
    root.destroy()

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            smile = False
            blink = False
            if Liveliness == True:
                cv2.putText(img, "Smile Detected, Liveliness Confirmed", (5,15), font, 1, (255,255,255), 2)

            if smile == False:
                import tkinter.messagebox as messagebox
                cv2.putText(img, "Smile for Liveliness Detection", (5,25), font, 1, (255,255,255), 2)

                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor= 1.5,
                    minNeighbors=15,
                    minSize=(25, 25),
                )
                for (xx, yy, ww, hh) in smile:
                    smile = True
                    cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
                
                    
                    cv2.putText(img, "Blink for Liveliness Detection", (5,35), font, 1, (255,255,255), 2)

                    eyes = eyeCascade.detectMultiScale(
                        roi_gray,
                        scaleFactor= 1.5,
                        minNeighbors=5,
                        minSize=(5, 5),
                        )

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]

                        _, eye_threshold = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
                        eye_height, eye_width = eye_threshold.shape[:2]
                        eye_area = eye_height * eye_width
                        eye_white_area = cv2.countNonZero(eye_threshold)
                        eye_ratio = eye_white_area / eye_area

                        if eye_ratio < 0.25 and smile == True:
                            blink_counter += 1
                            blink_detected = True
                            cv2.putText(roi_color, "Blink Detected", (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            blink_detected = False

                    if not blink_detected:
                        blink_counter = 0

                    if blink_counter >= blink_threshold:
                        Liveliness = True                       
                        LStr = 'Liveliness Detected, User: ' + id
                        show_alert(LStr)
                        sys.exit()




        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
