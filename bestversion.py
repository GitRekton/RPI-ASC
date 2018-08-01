import cv2
import numpy as np
import time
import math
import pigpio
from collections import deque
from thread import start_new_thread, allocate_lock
import smbus

#from __future__ import print_function
#from imutils.video.pivideostream import PiVideoStream
#from imutils.video import FPS

#-------INIT--------------------------------------------
time.sleep(0.3)
pi = pigpio.pi()

from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video import FPS
from imutils.video.pivideostream import PiVideoStream
time.sleep(0.3)


camera = PiCamera()
camera.resolution = (480, 640)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(480, 640))
stream = camera.capture_continuous(rawCapture, format="bgr",use_video_port=True)

lines = None
pi.set_PWM_dutycycle(12, 0)	#Sicherstellen, dass das Auto am Anfang still steht
#stream = PiVideoStream().start()


#FLAGS
RUNNING_ON_PI = True	#Abhaengig von der aktuellen Laufzeitumgebung
MOTOR_ACTIVE = False	#Debugvariable- aktiviert den Motor, waehrend Debug ausgeschaltet
BEEPER_ACTIVE = False	#Debugvariable- aktiviert den Beeper, waehrend Debug ausgeschaltet
THREAD_STARTED = False	#Flag fuer Mutlithreading

lock = allocate_lock()		#ermoeglicht eine Atomare Operation waehrend des Inits der Multithreading
Q = deque(4*[0], 4)		#Groesse des Schieberegisters, welches als Filter fungiert. Mittelwert des Registers = aktuelelr abstand zur naechsten Kurve
print "Init Queue"
Q_line0 = deque(4*[0], 4)
Q_line1 = deque(4*[0], 4)
Q_line2 = deque(4*[0], 4)
Q_line3 = deque(4*[0], 4)
###THREADING
num_threads = 0			#Thread counter

# FUNCTIONS FOR I2C-#######################FUNCTIONS FOR I2C-#######################FUNCTIONS FOR I2C-#######################FUNCTIONS FOR I2C-#######################
# Register
power_mgmt_1 = 0x6b	#i2c adresse
power_mgmt_2 = 0x6c	#i2c adresse

#bus = smbus.SMBus(1) # bus = smbus.SMBus(0) fuer Revision 1
address = 0x68       # via i2cdetect
 
# Aktivieren, um das Modul ansprechen zu koennen
#bus.write_byte_data(address, power_mgmt_1, 0)

#Thread um daten aus Sensor auszulesen
def get_acc_data(x):
        global num_threads, THREAD_STARTED
        lock.acquire()								#hier wird atomare Operation gestartet, um die naechsten Zeilen "in einem Rutsch" auszufuehren
        num_threads += 1
        THREAD_STARTED = True
        lock.release()								#hier endet die Atomare Operation.
        #hier wird der I2C ausgelesen
        while True:
                beschleunigung_xout = read_word_2c(0x3b)
                beschleunigung_yout = read_word_2c(0x3d)
                beschleunigung_zout = read_word_2c(0x3f)
                beschleunigung_xout_skaliert = beschleunigung_xout / 16384.0
                beschleunigung_yout_skaliert = beschleunigung_yout / 16384.0
                beschleunigung_zout_skaliert = beschleunigung_zout / 16384.0
                #print "beschleunigung_xout: ", ("%6d" % beschleunigung_xout), " skaliert: ", beschleunigung_xout_skaliert
                #print "beschleunigung_yout: ", ("%6d" % beschleunigung_yout), " skaliert: ", beschleunigung_yout_skaliert
                #print "beschleunigung_zout: ", ("%6d" % beschleunigung_zout), " skaliert: ", beschleunigung_zout_skaliert
                time.sleep(10)
        lock.acquire()
        num_threads -= 1
        lock.release()    
        return None
    
def read_byte(reg):
        return bus.read_byte_data(address, reg)

def get_median1(l):
	global Q
	Q.pop()
	Q.appendleft(l)
	med = []
	for elem in Q:
		med.append(elem)			
	return np.median(med)

def read_word(reg):
        h = bus.read_byte_data(address, reg)
        l = bus.read_byte_data(address, reg+1)
        value = (h << 8) + l
        return value
 
def read_word_2c(reg):
        val = read_word(reg)
        if (val >= 0x8000):
                return -((65535 - val) + 1)
        else:
                return val
 
def dist(a,b):
        return math.sqrt((a*a)+(b*b))
 
def get_y_rotation(x,y,z):
        radians = math.atan2(x, dist(y,z))
        return -math.degrees(radians)
 
def get_x_rotation(x,y,z):
        radians = math.atan2(y, dist(x,z))
        return math.degrees(radians)

#END FUNCTIONS FOR I2C######################END FUNCTIONS FOR I2C######################END FUNCTIONS FOR I2C######################END FUNCTIONS FOR I2C######################

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

def cv_colorspace(image_src):
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)	#LaneTracking
        #image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV) #LaneSideTracking with ColorFilter
        return image_src
    
def cv_img_filter(image_src):
	image_src = cv2.GaussianBlur(image_src, (7,7), 0)

def cv_threshold(image_src):
        #image_src2 = cv2.inRange(image_src, np.array([H_low,S_low,V_low]),np.array([H_high,S_high,V_high]))
        image_src = cv2.adaptiveThreshold(image_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 11, 2)	

def cv_img_transformation(image_src):
        #	pts1 = np.float32([[247,194],[312,194],[65,417],[532,417]])
        #	pts2 = np.float32([[0,0],[640,0],[0,480],[640,480]])
        #	M = cv2.getPerspectiveTransform(pts1, pts2)
        #	Minv = cv2.getPerspectiveTransform(pts2, pts1)
        #	image_src = cv2.warpPerspective(image_src, M, (640, 480))
        #image_src = cv2.warpPerspective(image_src, Minv, (640,480))
        return image_src

def cv_image_crop(image_src):
        #640, 480
        #print image_src.shape
        #					  Zeilen, 	Spalten
        #		[Horizont:Motorhaube, links:rechts]
        image_src = image_src[180:420, 180:430]	# [200:400, 250:300]	
        #image_src = cv2.resize(image_src, (0,0), image_src, fx=0.7, fy=0.7)
        return image_src

def cv_img_gradient(image_src):
        #sobelx.
        #image_src = cv2.Sobel(image_src, cv2.CV_8U, 1, 0, ksize=5)
        #sobely
        #image_src = cv2.Sobel(image_src, cv2.CV_8U, 0, 1, ksize=5)
        #image_src = cv2.Laplacian(image_src, cv2.CV_16U/cv2.CV_16S)	
        #	image_src = cv2.Canny(image_src, 100, 220)
        return image_src

def image_proc(image_src):
    #Image Colorspace
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)	#LaneTracking
        #image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV) #LaneSideTracking with ColorFilter

    #Image Crop
        #320,240
        #print image_src.shape[0]
        #					  Zeilen, 	Spalten
        #		[Horizont:Motorhaube, links:rechts]
        #image_src = image_src[int(image_src.shape[0] * 0.41):int(image_src.shape[0]), image_src.shape[1] * 0.4:image_src.shape[1] * 0.59]	# [200:400, 250:300]
        image_src = image_src[140:260, 120:240]
        image_src = cv2.GaussianBlur(image_src, (3,3), 0)
        #image_src = cv2.Laplacian(image_src, cv2.CV_8U)	
        #image_src = cv2.resize(image_src, (0,0), image_src, fx=0.7, fy=0.7)
        
    
    #Image Threshold
        #image_src2 = cv2.inRange(image_src, np.array([H_low,S_low,V_low]),np.array([H_high,S_high,V_high]))
        #image_src = cv2.Sobel(image_src, cv2.CV_8U, 1, 0, ksize=7)
        image_src = cv2.adaptiveThreshold(image_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 9, 2)
        image_src = np.invert(image_src)
        #image_src = cv2.Canny(image_src, 100, 150)
        return image_src


def get_median(l):
        global Q
        Q.pop()
        Q.appendleft(l)
        med = []
        for elem in Q:
                med.append(elem)			
        return np.median(med)

def trs(image_src):
	l, l_med, line_counter = 0,0,0
	#			image, -, -, threshold, maxLineGap, minLineLenght, 
	lines = cv2.HoughLinesP(image_src,1, np.pi/2, 10, 20, 60) # 2, 60)
	med_x1, med_x2, med_y1, med_y2 = 0,0,0,0
	if lines is not None:

                
                x1, x2, y1, y2 = [],[],[],[]
		for line in lines:
			try:
				coords = line[0]
                                line_counter += 1
				cv2.line(image_src, (coords[0], coords[1]), (coords[2], coords[3]), [100], 3)
                                #x1 = get_median_line0(coords[0])
				#x2 = get_median_line1(coords[1])
				#y1 = get_median_line2(coords[2])
				#y2 = get_median_line3(coords[3])
				
				#cv2.line(image_src, (coords[0], 240), (coords[2], coords[3]), [100], 7)
                                x1.append(coords[0])
                                x2.append(coords[1])
                                y1.append(coords[2])
                                y2.append(coords[3])
                                
                                
				#cv2.line(image_src, (0, 240 - int(l_med)),(250,240 - int(l_med)), [100], 9) 
				#print x1, y1, x2, y2
				#print coords
			except:
			    	pass
                #print x1 #= np.median(x1)
                #print x2 #= np.median(x2)
                #print y1 #= np.median(y1)
                #print np.median(y2) #= np.median(y2)
	
	#print (med_x1, med_x2, med_y1, med_y2), "MEDIAN for 1 pic"
	if lines is not None:
                #print line_counter, "line counter"
		l = absolute(142 - np.median(y2))
		l_med = get_median(l)
		if l_med > 180:
			set_motor_dutycycle(60)
			#speed = (0.5 * l_med)
			activate_beeper(1)
			#print l_med
		elif l_med < 180:
			set_motor_dutycycle(60)
			#speed = (0.2 * l_med)
			#print l_med
			activate_beeper(0)
#		set_motor_dutycycle(speed)
                if RUNNING_ON_PI == False:
                    cv2.line(image_src, (int(med_x1), int(med_x2)), (int(med_y1), int(med_y2)), [160], 20)
                    #cv2.line(image_src, (coords[0], 240), (coords[0], 240 - int(l_med)), [100], 20)
                    #cv2.line(image_src, (0, 240 - int(l_med)),(250,240 - int(l_med)), [100], 9) 

	else:
		speed = 60
		#print 0
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
                #image_src = imutils.resize(image_src, width=400)
		#cv2.namedWindow('image_src', WINDOW_NORMAL)
		cv2.imshow('image_src', image_src)
		
		
		cv2.moveWindow('image_src', 700, 700)
        
        
def get_median_line0(l):
	global Q_line0
	Q_line0.pop()
	Q_line0.appendleft(l)
	med0 = []
	for elem in Q:
		med0.append(elem)			
	return np.median(med0)

def get_median_line1(l):
	global Q_line1
	Q_line1.pop()
	Q_line1.appendleft(l)
	med1 = []
	for elem in Q:
		med1.append(elem)			
	return np.median(med1)

def get_median_line2(l):
	global Q_line2
	Q_line2.pop()
	Q_line2.appendleft(l)
	med2 = []
	for elem in Q:
		med2.append(elem)			
	return np.median(med2)

def get_median_line3(l):
	global Q_line3
	Q_line3.pop()
	Q_line3.appendleft(l)
	med3 = []
	for elem in Q:
		med3.append(elem)			
	return np.median(med3)

if __name__ == "__main__":
        print("[INFO] Init Cam")
        time.sleep(2.0)
        fps = FPS().start()
        for (i, f) in enumerate(stream):
                # grab the frame from the stream and resize it to have a maximum
                # width of 400 pixels
                frame = f.array
                #frame = imutils.resize(frame, width=400)
         
                # check to see if the frame should be displayed to our screen
                
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
         
                # clear the stream in preparation for the next frame and update
                # the FPS counter
                rawCapture.truncate(0)
                fps.update()
                break
                # check to see if the desired number of frames have been reached
         
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
         
        # do a bit of cleanup
        cv2.destroyAllWindows()
        stream.close()
        rawCapture.close()
        camera.close()

        # created a *threaded *video stream, allow the camera sensor to warmup,
        # and start the FPS counter
        print("[INFO] sampling THREADED frames from `picamera` module...")
        vs = PiVideoStream().start()
        time.sleep(1.0)
        fps = FPS().start()
        l = 0
        # loop over some frames...this time using the threaded stream
	perf = []
	if RUNNING_ON_PI == True:		        
            while True:
            #cap = cv2.VideoCapture('2018_06_20_2.h264')
            #	frame_counter = 0	
            #	init()
								#Dieser Block wird ausgefuehrt wenn das Programm von der Kamera Video beziehen soll
		##start_new_thread(get_acc_data, (None,))
		#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		while True:        
		        start1 = time.time()
		        #image_src = frame.array
                        image_src = vs.read()
                        cv2.imshow("Frame", image_src)
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
		        #rawCapture.truncate(0)      
		        bla = (end2-start2)*1000+(end3-start3)*1000
		        print (end1 - start1)*1000, "," , (end2-start2)*1000, "," , (end3-start3)*1000
		        
		        if cv2.waitKey(1) & 0xFF == ord('q'):
                                set_motor_dutycycle(0)
		                #cap.release()
	    #			cv2.destroyAllWindows()
                                stream.stop()
		                break
		        fps.update()
		        
	else:					#Dieser Block wird ausgefuehrt, wenn das Programm von einer Datei das Video lesen soll
                cap = cv2.VideoCapture('2018_07_02_1.h264')
		while True:
			start1 = time.time()		
			ret, image_src = cap.read()
			end1 = time.time()
			cv2.imshow("test", image_src)
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
