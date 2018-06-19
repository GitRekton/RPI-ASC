import cv2
import numpy as np
import time
import math
from collections import deque

time.sleep(0.3)

lines = None

#FLAGS
RUNNING_ON_PI = False	#Abhaengig von der aktuellen Laufzeitumgebung

Q = deque(4*[0], 4)

def set_motor_dutycycle(x):
	global RUNNING_ON_PI
	if RUNNING_ON_PI == True:
		pi.set_PWM_dutycycle(12, x)
	else:
		pass

def activate_beeper(x):					# Parameter: 1 = An; 0 = Aus
	global RUNNING_ON_PI
	if RUNNING_ON_PI == True:
		global pi
		if x == 1:
			pi.set_PWM_dutycycle(18, 120)
		else:
			pi.set_PWM_dutycycle(18, 0)

def init_for_Pi():
	import pigpio
	time.sleep(0.3)
	pi = pigpio.pi()
	from picamera.array import PiRGBArray
	from picamera import PiCamera
	time.sleep(0.3)
	camera = PiCamera()
	camera.resolution = (640, 480) #640,480
	camera.framerate = 15#
	rawCapture = PiRGBArray(camera, size=(640, 480))
	
	lines = None

	time.sleep()

	set_motor_dutycycle(0)	#Sicherstellen, dass das Auto am Anfang still steht
	
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
	cv2.namedWindow('image2')
	cv2.createTrackbar('cb', 'image2', 0, 300, nothing)
	cv2.createTrackbar('ct', 'image2', 0, 300, nothing)
	cv2.createTrackbar('H_low', 'image2', 0, 255, nothing)
	cv2.createTrackbar('H_high', 'image2', 0, 255, nothing)
	cv2.createTrackbar('S_low', 'image2', 0, 255, nothing)
	cv2.createTrackbar('S_high', 'image2', 0, 255, nothing)
	cv2.createTrackbar('V_low', 'image2', 0, 255, nothing)
	cv2.createTrackbar('V_high', 'image2', 0, 255, nothing)


def image_proc(image_src):
	image_src2 = image_src
	image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)	#LaneTracking
	image_src2 = cv2.cvtColor(image_src2, cv2.COLOR_BGR2HSV) #LaneSideTracking with ColorFilter
	#cv2.imshow("org", image_src)
	#cv2.imshow("org2", image_src2)
	H_low = cv2.getTrackbarPos('H_low', 'image2')
	H_high = cv2.getTrackbarPos('H_high', 'image2')
	S_low = cv2.getTrackbarPos('H_low', 'image2')
	S_high = cv2.getTrackbarPos('H_high', 'image2')
	V_low = cv2.getTrackbarPos('H_low', 'image2')
   	V_high = cv2.getTrackbarPos('H_high', 'image2')

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
		

	image_src = cv2.GaussianBlur(image_src, (5,5), 0)
	image_src = cv2.medianBlur(image_src, 5)	
	cb = cv2.getTrackbarPos('cb', 'image2')
	ct = cv2.getTrackbarPos('ct', 'image2')
		

	image_src = cv2.Sobel(image_src, cv2.CV_8U, 1, 0, ksize=5)
	ret, image_src = cv2.threshold(image_src, 127, 255, 0)
	ret, image_src2 = cv2.threshold(image_src, 127, 255, 2)
	ret, image_src3 = cv2.threshold(image_src, 127, 255, 3)

#	sobelx = cv2.inRange(sobelx, np.array([160]),np.array([255]))
#	sobely = cv2.Sobel(image_src, cv2.CV_8U, 0, 1, ksize=5)
	cv2.imshow("sobelx", image_src)
	cv2.imshow("sobelx2", image_src2)
	cv2.imshow("sobelx3", image_src3)
#	cv2.moveWindow('sobelx', 380, 700)
#	cv2.imshow("sobely", sobely)
#	cv2.moveWindow('sobelx', 380, 700)
#	image_src = cv2.Canny(image_src,cb,ct)
#	cv2.imshow("canny", image_src)
  
			
	return image_src
#	return sobelx	



def trs(image_src):
	l = 0
        global Q
	

	#			image, -, -, threshold, maxLineGap, minLineLenght, 
	lines = cv2.HoughLinesP(image_src,1, np.pi, 100, 60, 100) # 2, 60)
	if lines is not None:
		for line in lines:
			try:
				coords = line[0]
				cv2.line(image_src, (coords[0], coords[1]), (coords[2], coords[3]), [255], 7)

				cv2.line(image_src, (coords[0], 240), (coords[2], coords[3]), [100], 7)

				cv2.line(image_src, (0, coords[3]),(250,coords[3]), [100], 9) 
				#print "x1, y1, x2, y2"
				#print coords
			except:
			    	pass	
	if lines is not None:
		l = absolute(240 - coords[3])
		Q.pop()
		Q.appendleft(l)
		print Q
		med = []
		for elem in Q:
			med.append(elem)			
		print np.median(med)
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
		print 0
		set_motor_dutycycle(60)
		activate_beeper(0)
	return l
        
def image_display(image_src, lines,l):#l
        font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image_src, str(l), (3,30), font, 0.5, (255,255,255), 2, 0)
	#if lines is None:
	#	cv2.putText(image_src, 'Curve', (3,30), font, 0.5, (255,255,255), 2, 0)
	#		
	#else:
	#	cv2.putText(image_src, str(l) , (3,30), font, 0.5, (255,255,255), 2, 0)
			
	if image_src is not None:
		cv2.imshow('image_src', image_src)
		cv2.moveWindow('image_src', 700, 700)
        
if __name__ == "__main__":
	cap = cv2.VideoCapture('testfahrtnew2.h264')
	frame_counter = 0	
	init()
	while True:
		
		#image_src = frame.array
		start1 = time.time()
		ret, image_src = cap.read()
		end1 = time.time()

		#start2 = time.time()
		image_src = image_proc(image_src)
		#end2 = time.time()
                
		#start3 = time.time()
		l = trs(image_src)
		#end3 = time.time()
		#out.write(image_src)

		#start4 = time.time()
		image_display(image_src,lines, l) #cap,l
		#end4 = time.time()
		#rawCapture.truncate(0)      
		#print ("ImageProc ", ((end2-start2)*1000),"TRS ", ((end3-start3)*1000), "Display ", (end4 - start4))
		if cv2.waitKey(600) & 0xFF == ord('q'):
			set_motor_dutycycle(0)
			cap.release()
			cv2.destroyAllWindows()
			break 	


