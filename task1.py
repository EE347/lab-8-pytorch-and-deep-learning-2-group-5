from picamera2 import Picamera2, Preview
import time
import cv2

recording = False
num = 0
numimg = 0
numimgvid = 0
output = None

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":'XRGB8888', "size": (640, 480)}))
picam2.start()
while True:
    img = picam2.capture_array()
    greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(greyimg, 1.1, 9)
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
  
    if(key == ord("q")):
        break
    if(key == ord("c")):
        numimg=0
        while (numimg < 60):
            #image_filename = "/home/pi/ee347/lab-6-python-and-opencv-2-group-5/task10"+str(numimg)+".jpg"
            cropped_image_filename=""
            if (numimg <50):
                cropped_image_filename = "/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/data/train/0/pic"+str(numimg)+".jpg"
            else:
                cropped_image_filename = "/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/data/test/0/pic"+str(numimg)+".jpg"
            #cv2.imwrite(image_filename, img)
            greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_rect = haar_cascade.detectMultiScale(greyimg, 1.1, 9)
            for (x, y, w, h) in faces_rect:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_region = img[y:y+h, x:x+w]
                face = cv2.resize(face_region, (64,64))
                cv2.imwrite(cropped_image_filename, face_region)
        
            numimg = numimg+1


    
    if(key == ord("d")):
        numimg = 0
        while (numimg < 60):
            #image_filename = "/home/pi/ee347/lab-6-python-and-opencv-2-group-5/task10"+str(numimg)+".jpg"
            cropped_image_filename=""
            if (numimg <50):
                cropped_image_filename = "/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/data/train/1/pic"+str(numimg)+".jpg"
            else:
                cropped_image_filename = "/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-5/data/test/1/pic"+str(numimg)+".jpg"
            #cv2.imwrite(image_filename, img)
            greyimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_rect = haar_cascade.detectMultiScale(greyimg, 1.1, 9)
            for (x, y, w, h) in faces_rect:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_region = img[y:y+h, x:x+w]
                face = cv2.resize(face_region, (64,64))
                cv2.imwrite(cropped_image_filename, face)
        
            numimg = numimg+1

cv2.destroyAllWindows()