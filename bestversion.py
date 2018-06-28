import cv2
import numpy as np
import time
import math
import pigpio
from collections import deque

time.sleep(0.3)
pi = pigpio.pi()
from picamera.array import PiRGBArray
from picamera import PiCamera
time.sleep(0.3)
camera = PiCamera()
camera.resolution = (640, 480) #640,480
camera.framerate = 60#
rawCapture = PiRGBArray(camera, size=(640, 480))
lines = None
set_motor_dutycycle(0)	#Sicherstellen, dass das Auto am Anfang still steht

#FLAGS
RUNNING_ON_PI = True	#Abhaengig von der aktuellen Laufzeitumgebung
MOTOR_ACTIVE = False
BEEPER_ACTIVE = False

Q = deque(4*[0], 4)

def set_motor_dutycycle(x):
	global RUNNING_ON_PI
	if RUNNING_ON_PI == True and MOTOR_ACTIVE == True:
		pi.set_PWM_dutycycle(12, x)
	else:
		pass

def activate_beeper(x):					# Parameter: 1 = An; 0 = Aus
	global RUNNING_ON_PI
	if RUNNING_ON_PI == True and BEEPER_ACTIVE == True:
		global pi
		if x == 1:
			pi.set_PWM_dutycycle(18, 120)
		else:
			pi.set_PWM_dutycycle(18, 0)


for i in range(4):
        activate_beeper(1)
        time.sleep(0.1)
        activate_beeper(0)
        time.sleep(0.1)

def absolute(x):
	return -x if x < 0 else x

def nothing(x):
	pass
    
def init():
	max_line_gap = 0
	min_line_lenght = 0
#	cv2.namedWindow('image2')
	cv2.createTrackbar('cb', 'image2', 0, 300, nothing)
	cv2.createTrackbar('ct', 'image2', 0, 300, nothing)
	cv2.createTrackbar('H_low', 'image2', 0, 179, nothing)
	cv2.createTrackbar('H_high', 'image2', 0, 179, nothing)
	cv2.createTrackbar('S_low', 'image2', 0, 255, nothing)
	cv2.createTrackbar('S_high', 'image2', 0, 255, nothing)
	cv2.createTrackbar('V_low', 'image2', 0, 255, nothing)
	cv2.createTrackbar('V_high', 'image2', 0, 255, nothing)


def image_proc(image_src):
	#image_src2 = image_src
	image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)	#LaneTracking
	#image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV) #LaneSideTracking with ColorFilter
	#cv2.imshow("org2", image_src2)
#	H_low = cv2.getTrackbarPos('H_low', 'image2')
#	H_high = cv2.getTrackbarPos('H_high', 'image2')
#	S_low = cv2.getTrackbarPos('H_low', 'image2')
#	S_high = cv2.getTrackbarPos('H_high', 'image2')
#	V_low = cv2.getTrackbarPos('H_low', 'image2')
 #  	V_high = cv2.getTrackbarPos('H_high', 'image2')

	#image_src2 = cv2.inRange(image_src2, np.array([H_low,S_low,V_low]),np.array([H_high,S_high,V_high]))
	#cv2.imshow("org3", image_src2)
#	pts1 = np.float32([[247,194],[312,194],[65,417],[532,417]])
#	pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
#	M = cv2.getPerspectiveTransform(pts1, pts2)
#	Minv = cv2.getPerspectiveTransform(pts2, pts1)

#	image_src = cv2.warpPerspective(image_src, M, (640, 480))
	#image_src = cv2.warpPerspective(image_src, Minv, (640,480))
#	cv2.imshow("trans", image_src)		

	# 640, 480
	#print image_src.shape
	#lines = None
   	#					  Zeilen, 	Spalten
	#		[Horizont:Motorhaube, links:rechts]
	image_src = image_src[180:420, 180:430]	# [200:400, 250:300]	
   	#image_src = cv2.resize(image_src, (0,0), image_src, fx=0.7, fy=0.7)
   	#print image_src.shape
#	cv2.imshow("org", image_src)

	image_src = cv2.GaussianBlur(image_src, (5,5), 0)
	#image_src = cv2.Laplacian(image_src, cv2.CV_16U/cv2.CV_16S)	
	
	#cb = cv2.getTrackbarPos('cb', 'image2')
	#ct = cv2.getTrackbarPos('ct', 'image2')
		

	image_src = cv2.Sobel(image_src, cv2.CV_16U/cv2.CV_16S, 1, 0, ksize=5)
	#ret, image_src = cv2.threshold(image_src, [0,0,0], [0,0,255], 0)
	#image_src = cv2.inRange(image_src, np.array([H_low]),np.array([H_high]))
	#image_src = cv2.Sobel(image_src, cv2.CV_8U, 1, 0, ksize=5)
#	image_src = cv2.Canny(image_src, 100, 220)
#	cv2.imshow("Laplacian", image_src)
#	cv2.moveWindow('sobelx', 380, 700)

#	cv2.moveWindow('sobelx', 380, 700)
#	image_src = cv2.Canny(image_src,cb,ct)
#	cv2.imshow("canny", image_src)
  
			
	return image_src
#	return sobelx	


def get_median(l):
	global Q
	Q.pop()
	Q.appendleft(l)
	med = []
	for elem in Q:
		med.append(elem)			
	return np.median(med)

def trs(image_src):
	l, l_med = 0,0
	
        
	

	#			image, -, -, threshold, maxLineGap, minLineLenght, 
	lines = cv2.HoughLinesP(image_src,1, np.pi, 160, 70, 100) # 2, 60)
	if lines is not None:
		for line in lines:
			try:
				coords = line[0]
				#cv2.line(image_src, (coords[0], coords[1]), (coords[2], coords[3]), [255], 7)

				#cv2.line(image_src, (coords[0], 240), (coords[2], coords[3]), [100], 7)

				#cv2.line(image_src, (0, 240 - int(l_med)),(250,240 - int(l_med)), [100], 9) 
				#print "x1, y1, x2, y2"
				#print coords
			except:
			    	pass	
	if lines is not None:
		l = absolute(240 - coords[3])
		l_med = get_median(l)
		if l_med > 180:
			set_motor_dutycycle(60)
			#speed = (0.5 * l_med)
			activate_beeper(1)
			print l_med
		elif l_med < 180:
			set_motor_dutycycle(60)
			#speed = (0.2 * l_med)
			print l_med
			activate_beeper(0)
#		set_motor_dutycycle(speed)
                if RUNNING_ON_PI == False:
                    cv2.line(image_src, (coords[0], 240), (coords[0], 240 - int(l_med)), [100], 20)
                    cv2.line(image_src, (0, 240 - int(l_med)),(250,240 - int(l_med)), [100], 9) 
#		if l > 170:
#			print l
#		    	set_motor_dutycycle(70)
#		    	activate_beeper(1)
#		else:
#			print l
#			set_motor_dutycycle(30)
#			activate_beeper(1)
 #                       return 0
	else:
		speed = 60
		print 0
		set_motor_dutycycle(speed)
		activate_beeper(0)
	return l_med
        
def image_display(image_src, lines,l):#l
        font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image_src, str(l), (3,30), font, 0.7, (120), 2, 0)
	if lines is None:
		cv2.putText(image_src, 'Curve', (3,30), font, 0.5, (255,255,255), 2, 0)
			
	else:
		cv2.putText(image_src, str(l) , (3,30), font, 0.5, (255,255,255), 2, 0)
			
	if image_src is not None:
		cv2.imshow('image_src', image_src)
		cv2.moveWindow('image_src', 700, 700)
        
if __name__ == "__main__":
	#cap = cv2.VideoCapture('2018_06_20_2.h264')
#	frame_counter = 0	
#	init()
	l = 0
	perf = []
	if RUNNING_ON_PI == True:
            for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                    start1 = time.time()		
                    image_src = frame.array
                    end1 = time.time()

                    start2 = time.time()
                    image_src = image_proc(image_src)
                    end2 = time.time()
                    
                    start3 = time.time()
                    l = trs(image_src)
                    end3 = time.time()
                    #out.write(image_src)

                    start4 = time.time()
                    image_display(image_src,lines, l) #cap,l
                    end4 = time.time()
                    rawCapture.truncate(0)      
                    bla = (start1-end1)*1000+(end2-start2)*1000+(end3-start3)*1000
                    perf.append(bla)
                    print np.median(perf)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            set_motor_dutycycle(0)
                            #cap.release()
        #			cv2.destroyAllWindows()
                            break 	
        else:
            cap = cv2.VideoCapture('2018_06_20_2.h264')
            while True:
                start1 = time.time()		
                ret, image_src = cap.read()
                end1 = time.time()
                
                start2 = time.time()
                image_src = image_proc(image_src)
                end2 = time.time()
                
                start3 = time.time()
                l = trs(image_src)
                end3 = time.time()

    		start4 = time.time()
    		image_display(image_src,lines, l) #cap,l
    		end4 = time.time()     
    		bla = (start1-end1)*1000+(end2-start2)*1000+(end3-start3)*1000
                perf.append(bla)
                print np.median(perf)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        set_motor_dutycycle(0)
                        #cap.release()
#			cv2.destroyAllWindows()
                        break 	
