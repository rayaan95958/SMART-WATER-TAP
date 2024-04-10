import cv2
from scipy.spatial import distance as dist
from cvzone.FaceDetectionModule import FaceDetector
from imutils.video import VideoStream
from imutils import face_utils
from threading import Timer
import pyfirmata2
import numpy as np
import argparse
import imutils
import time
import dlib


#To stop mouth detection when spraying.
Flag=0 

#Arduino-setup
port = "COM14"
board = pyfirmata2.Arduino(port)
iter8 = pyfirmata2.util.Iterator(board)
iter8.start()

#Servo-setup
servo_pinX = board.get_pin('d:8:s') #Servo-x to pin 9 
servo_pinY = board.get_pin('d:9:s') #Servo-y to pin 10         
servoPos = [90, 90] # initial servo position

#Led-setup
led_pinG=board.get_pin('d:11:o') 
led_pinB=board.get_pin('d:12:o')

#Buzzer-setup
Buzzer=board.get_pin('d:6:p')

#Pump-setup
EN1=board.get_pin('d:3:p')
IN1=board.get_pin('d:5:o')
IN2=board.get_pin('d:4:o')
IN1.write(0)
IN2.write(1)

#Arduino-functions
def buzzer_intro():
    Buzzer.write(0.1)
    time.sleep(0.3)
    Buzzer.write(0.0)

def buzzer_outro():
    Buzzer.write(0.4)
    time.sleep(0.3)
    Buzzer.write(0.0)

def servo_calc():
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]
        servoX = np.interp(fx, [0, frame_width], [0, 180])
        servoY = np.interp(fy, [0, frame_height], [0, 180])

        if servoX < 0:
            servoX = 0
        elif servoX > 180:
            servoX = 180
        if servoY < 0:
            servoY = 0
        elif servoY > 90:   #Any angle under this is impractical
            servoY = 90

        servoPos[0] = servoX
        servoPos[1] = servoY       

def move_servoX(): 
        servo_pinX.write(servoPos[0])

def move_servoY():
        servo_pinY.write(servoPos[1])      

def pump_on():
    led_on("BLUE")
    EN1.write(0.8)  
    time.sleep(2)
    pump_off()

def pump_off():
    EN1.write(0)
    buzzer_outro()
    led_on("GREEN")

def led_on(Colour):
    if(Colour=="GREEN"):
       led_pinB.write(1)
       led_pinG.write(0)
       time.sleep(3)
       global Flag
       Flag=0

    elif(Colour=="BLUE"):
        led_pinG.write(1)
        led_pinB.write(0)

###################################################################################################
    
#Mouth pre-declaration
def mouth_aspect_ratio(mouth): 
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
                help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.90

print("[INFO] loading facial landmark predictor...")
detector_mouth = dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor(args["shape_predictor"])
# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

#Input
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
frame_width =640
frame_height =360

#Output 
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
time.sleep(1.0)

detector_face=FaceDetector()

#Active loop
while True:
    temp_img = vs.read()
    img = cv2.flip(temp_img,1)
    img,bboxs = detector_face.findFaces(img, draw=False)
    img = imutils.resize(img, width=640)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector_mouth(gray, 0)	

    if bboxs:
        #Get the coordinates
        fx, fy = bboxs[0]["center"][0], bboxs[0]["center"][1]
        pos = [fx, fy]
        cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
        cv2.putText(img, str(pos), (fx+15, fy-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2 )
        cv2.line(img, (0, fy), (frame_width, fy), (0, 0, 0), 2)  # x line
        cv2.line(img, (fx, frame_height), (fx, 0), (0, 0, 0), 2)  # y line
        cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(img,"TARGET LOCKED", (850, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3 )

        #Calculate servo angle
        if(Flag==0):
            servo_calc()

    else:
        cv2.putText(img,"NO TARGET", (880, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.circle(img, (640, 360), 80, (0, 0, 255), 2)
        cv2.circle(img, (640, 360), 15, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (0, 360), (frame_width, 360), (0, 0, 0), 2)  # x line
        cv2.line(img, (640, frame_height), (640, 0), (0, 0, 0), 2)  # y line
    
    cv2.putText(img, f'Servo X: {int(servoPos[0])} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Servo Y: {int(servoPos[1])} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    move_servoX()
    move_servoY()

    #Mouth-open detection
    if(Flag==0):
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth = shape[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(img, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mar > MOUTH_AR_THRESH:
                cv2.putText(img, "Mouth is Open!", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                Flag=1
                buzzer_intro()
                Timer(1,pump_on).start()
                
            out.write(img)
   
    cv2.imshow("Image",img)
    
    #Exit command
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
