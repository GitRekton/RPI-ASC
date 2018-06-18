import cv2
import numpy as np
import time
import math

time.sleep(0.3)

lines = None

def init_for_Pi():
	import pigpio
	time.sleep(0.3)
	from picamera.array import PiRGBArray
	from picamera import PiCamera
	time.sleep(0.3)
	camera = PiCamera()
	camera.resolution = (640, 480) #640,480
	camera.framerate = 15#
	rawCapture = PiRGBArray(camera, size=(640, 480))
	
	lines = None

	pi = pigpio.pi()
	time.sleep()

	pi.set_PWM_dutycycle(12, 0)
	for i in range(4):
		pi.set_PWM_dutycycle(18, 80)
		time.sleep(0.1)
		pi.set_PWM_dutycycle(18, 0)
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
	cv2.createTrackbar('H_low', 'image2', 0, 400, nothing)
	cv2.createTrackbar('H_high', 'image2', 0, 400, nothing)
	cv2.createTrackbar('S_low', 'image2', 0, 100, nothing)
	cv2.createTrackbar('S_high', 'image2', 0, 100, nothing)
	cv2.createTrackbar('V_low', 'image2', 0, 100, nothing)
	cv2.createTrackbar('V_high', 'image2', 0, 100, nothing)

def image_proc(image_src):
	image_src2 = image_src
	image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
	image_src2 = cv2.cvtColor(image_src2, cv2.COLOR_BGR2HSV)
	cv2.imshow("org", image_src)
	cv2.imshow("org2", image_src2)
	H_low = cv2.getTrackbarPos('H_low', 'image2')
	H_high = cv2.getTrackbarPos('H_high', 'image2')
	S_low = cv2.getTrackbarPos('H_low', 'image2')
	S_high = cv2.getTrackbarPos('H_high', 'image2')
	V_low = cv2.getTrackbarPos('H_low', 'image2')
   	V_high = cv2.getTrackbarPos('H_high', 'image2')

	image_src2 = cv2.inRange(image_src2, np.array([H_low,S_low,V_low]),np.array([H_high,S_high,V_high]))
	cv2.imshow("org3", image_src2)
	#pts1 = np.float32([[247,194],[312,194],[65,417],[532,417]])
	#pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
	#M = cv2.getPerspectiveTransform(pts1, pts2)
	#Minv = cv2.getPerspectiveTransform(pts2, pts1)

	#image_src = cv2.warpPerspective(image_src, M, (640, 480))
	#image_src = cv2.warpPerspective(image_src, Minv, (640,480))
	#cv2.imshow("trans", image_src)		

	# 640, 480
	#print image_src.shape
	#lines = None
   	#					  Zeilen, 	Spalten
	#		[Horizont:Motorhaube, links:rechts]
	image_src = image_src[10:400, 180:430]	# [200:400, 250:300]	
   	#image_src = cv2.resize(image_src, (0,0), image_src, fx=0.7, fy=0.7)
   	#print image_src.shape
		

	image_src = cv2.GaussianBlur(image_src, (5,5), 0)
	cv2.imshow("gauss", image_src)
		
	cb = cv2.getTrackbarPos('cb', 'image2')
	ct = cv2.getTrackbarPos('ct', 'image2')
		
	#sobelx = cv2.Sobel(image_src, cv2.CV_64F, 1, 0, ksize=3)
	#cv2.imshow("Sobel", sobelx)
	image_src = cv2.Canny(image_src,80,88)
	##cv2.imshow("canny", image_src)
  
			
	return image_src

def trs(image_src):
        l = 0
        

    	#			image, -, -, threshold, minLineLenght, maxLineGap
        lines = cv2.HoughLinesP(image_src,1, np.pi, 180, 100, 120) # 2, 60)
        if lines is not None:
                for line in lines:
                        try:
                                coords = line[0]
				cv2.line(image_src, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 7)
				#print "x1, y1, x2, y2"
				#print coords
			except:
			    	pass	
	if lines is not None:
		l = absolute(coords[1] - coords[3])
		print l
		if l > 190:
			pass	
		elif l > 169 and l < 190:
			print l
#		    	pi.set_PWM_dutycycle(12,70)
#		    	pi.set_PWM_dutycycle(18,120)
		else:
			print "Curve ahead"
#			pi.set_PWM_dutycycle(12,30)
#			pi.set_PWM_dutycycle(18,120)
                        return 0
	else:
		print "Curve"
#		pi.set_PWM_dutycycle(12,60)
#		pi.set_PWM_dutycycle(18,0)
	return l
        
def image_display(image_src, lines):#l
        font = cv2.FONT_HERSHEY_SIMPLEX
	#if lines is None:
	#	cv2.putText(image_src, 'Curve', (3,30), font, 0.5, (255,255,255), 2, 0)
	#		
	#else:
	#	cv2.putText(image_src, str(l) , (3,30), font, 0.5, (255,255,255), 2, 0)
			
	if image_src is not None:
		cv2.imshow('image_src', image_src)
        
def main():
	cap = cv2.VideoCapture('testfahrtnew2.h264')
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
		image_display(image_src,lines) #cap,l
		#end4 = time.time()
		#rawCapture.truncate(0)      
		#print ("ImageProc ", ((end2-start2)*1000),"TRS ", ((end3-start3)*1000), "Display ", (end4 - start4))
		if cv2.waitKey(1) & 0xFF == ord('q'):
                        #pi.set_PWM_dutycycle(12,0)
                        cv2.destroyAllWindows()
			break 	

main()
