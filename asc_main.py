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

DIM=(640,480)
K=np.array([[333.3095682593701, 0.0, 299.42451764331906], [0.0, 333.39606366460384, 227.39887052277433], [0.0, 0.0, 1.0]])
D=np.array([[-0.057478717587506466], [0.10222798530534297], [-0.13515780465585153], [0.05446853521315723]])

camera = PiCamera()
camera.resolution = (480, 640)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(480, 640))
stream = camera.capture_continuous(rawCapture, format="bgr",use_video_port=True)

lines = None
pi.set_PWM_dutycycle(12, 0)	#Sicherstellen, dass das Auto am Anfang still steht
#stream = PiVideoStream().start()


#FLAGS
RUNNING_ON_PI = False	#Abhaengig von der aktuellen Laufzeitumgebung
MOTOR_ACTIVE = False	#Debugvariable- aktiviert den Motor, waehrend Debug ausgeschaltet
BEEPER_ACTIVE = False	#Debugvariable- aktiviert den Beeper, waehrend Debug ausgeschaltet
THREAD_STARTED = False	#Flag fuer Mutlithreading

straight_flag = True
curve_flage = False
curve_is_over_flag = False
            
lock = allocate_lock()		#ermoeglicht eine Atomare Operation waehrend des Inits der Multithreading
Q = deque(4*[0], 4)		#Groesse des Schieberegisters, welches als Filter fungiert. Mittelwert des Registers = aktuelelr abstand zur naechsten Kurve
print "Init Queue"

###THREADING
num_threads = 0			#Thread counter

# FUNCTIONS FOR I2C-#######################FUNCTIONS FOR I2C-#######################FUNCTIONS FOR I2C-#######################FUNCTIONS FOR I2C-#######################
# Register
power_mgmt_1 = 0x6b	#i2c adresse
power_mgmt_2 = 0x6c	#i2c adresse

bus = smbus.SMBus(1) # bus = smbus.SMBus(0) fuer Revision 1
address = 0x68       # via i2cdetect
 
# Aktivieren, um das Modul ansprechen zu koennen
#bus.write_byte_data(address, power_mgmt_1, 0)

#Thread um daten aus Sensor auszulesen
acc_data = np.array([0,0,0,0,0,0])

def get_acc_data(x):
        global acc_data, num_threads, THREAD_STARTED
        lock.acquire()								#hier wird atomare Operation gestartet, um die naechsten Zeilen "in einem Rutsch" auszufuehren
        num_threads += 1
        THREAD_STARTED = True
        lock.release()								#hier endet die Atomare Operation.
        #hier wird der I2C ausgelesen
        while True:
                beschleunigung_xout = read_word_2c(0x3b)
                beschleunigung_yout = read_word_2c(0x3d)
                beschleunigung_zout = read_word_2c(0x3f)
                gyroskop_xout = read_word_2c(0x43)
                gyroskop_yout = read_word_2c(0x45)
                gyroskop_zout = read_word_2c(0x47)
                beschleunigung_xout_skaliert = beschleunigung_xout / 16384.0
                beschleunigung_yout_skaliert = beschleunigung_yout / 16384.0
                beschleunigung_zout_skaliert = beschleunigung_zout / 16384.0
                gyroskop_xout = gyroskop_xout / 131
                gyroskop_yout = gyroskop_yout / 131
                gyroskop_zout = gyroskop_zout / 131
                #print "beschleunigung_xout: ", ("%6d" % beschleunigung_xout), " skaliert: ", beschleunigung_xout_skaliert
                #print "beschleunigung_yout: ", ("%6d" % beschleunigung_yout), " skaliert: ", beschleunigung_yout_skaliert
                #print "beschleunigung_zout: ", ("%6d" % beschleunigung_zout), " skaliert: ", beschleunigung_zout_skaliert
                #print beschleunigung_xout_skaliert,",", beschleunigung_yout_skaliert,",", beschleunigung_zout,",", gyroskop_xout,",", gyroskop_yout,",", gyroskop_zout
                acc_data = np.array([beschleunigung_xout, beschleunigung_yout, beschleunigung_zout, gyroskop_xout, gyroskop_yout, gyroskop_zout])
                time.sleep(0.05)
        lock.acquire()
        num_threads -= 1
        lock.release()    
        return None

def gyro_regelung(x):            #DAS IST SHIT, morgen wieder rasnehmen
        global acc_data, curve_flag, curve_is_over_flag, straight_flag
        pi.write(6, 0)
        time.sleep(0.005)
	gier = acc_data[5]
        while True:
            straight_flag = True
            curve_flage = False
            curve_is_over_flag = False
            time.sleep(0.005)
	    gier = acc_data[5]
            while gier > 110:
                curve_flag = True
                curve_is_over_flag = False
                straight_flag = False
                time.sleep(0.005)
                gier =  acc_data[5]
                if gier < 87:
                    curve_is_over_flag = True
                    curve_flag = False
                    straight_flag = False
                    time.sleep(0.26)
		    break
	return 0
                            
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
		pi.set_PWM_dutycycle(18, x)
	else:
		pass

def activate_beeper(x):					# Parameter: 1 = An; 0 = Aus
	global RUNNING_ON_PI
	if RUNNING_ON_PI == True and BEEPER_ACTIVE == True:
		global pi
		if x == 1:
			pi.set_PWM_dutycycle(18, 40)
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
def draw_grid(image_src):  #Funktion um Gitternetzlinien auf Bild zu zeichnen
        i = 0
        while i < 640:
                cv2.line(image_src, (i,0),(i,image_src.shape[0]), [100], 2)
                i += 25
        j = 0
        while j < 480:
                cv2.line(image_src, (0,j),(image_src.shape[1],j), [100], 2)
                j += 25
        return image_src
    
def image_proc(image_src):
    #Image Colorspace
                            #[240:480,0:640]
        s1 = time.time()
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        #image_src = undistort(image_src)  #ist erstmal raus, braucht zu lange
        image_src = image_src[260:400,0:640]  #image_src[240:400,0:640]
        
        cv2.imshow("cropped", image_src)
     
        graybot = cv2.getTrackbarPos("bot", "slider") #0
        graytop = cv2.getTrackbarPos("top", "slider")   #193

        e1 = time.time()
        
        s2 = time.time()
        #Transformation
        src = np.float32([[0,140], [0,0],[640,0],[640,140]])  #np.float32([[0,160], [0,0],[640,0],[640,160]])
        dst = np.float32([[280,140], [0,0],[640,0],[360,138]])  #np.float32([[280,160], [0,0],[640,0],[360,160]])
        M = cv2.getPerspectiveTransform(src, dst)
        
        top_view = cv2.warpPerspective(image_src,  M, (640,140))
        
        cv2.imshow("ok", top_view)
        
        top_view = top_view[:,260:380]
        top_view = cv2.GaussianBlur(top_view, (5,5), 0)
        cv2.imshow("top_view", cv2.Canny(top_view,100,220))
        top_view_sobel = cv2.Sobel(top_view, cv2.CV_8U, 1, 0, ksize=3)  #cv2.Sobel(top_view, cv2.CV_8U, 1, 0, ksize=3)
        print top_view_sobel.shape
        cv2.imshow("top_view_sobel", top_view_sobel)
        cv2.imshow("top_view_THSOBEL", np.invert(cv2.adaptiveThreshold(top_view_sobel,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 3, 8)))
        
        top_view_gray = cv2.inRange(top_view, 200, 255)

        top_view = cv2.adaptiveThreshold(top_view, 255, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 7, 8)
        e2 = time.time()
        top_view = np.invert(top_view)

        #top_view = cv2.resize(top_view, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        
        cv2.imshow("gray", top_view)
        cv2.imshow("whitefilter", top_view_gray)
        #Image Crop
        #320,240
        #print image_src.shape[0]
        #					  Zeilen, 	Spalten
        #		[Horizont:Motorhaube, links:rechts]
        #image_src = image_src[int(image_src.shape[0] * 0.41):int(image_src.shape[0]), image_src.shape[1] * 0.4:image_src.shape[1] * 0.59]	# [200:400, 250:300]
        #image_src = image_src[140:260, 120:240]
        
        #image_src = cv2.Laplacian(image_src, cv2.CV_8U)	
        #image_src = cv2.resize(image_src, (0,0), image_src, fx=0.7, fy=0.7)
        
    
        #Image Threshold
        #top_view = cv2.Canny(top_view, 100, 150)
        #image_src2 = cv2.inRange(image_src, np.array([H_low,S_low,V_low]),np.array([H_high,S_high,V_high]))
        #top_view = cv2.Sobel(top_view, cv2.CV_8U, 1, 0, ksize=7)
        #image_src_straight = cv2.adaptiveThreshold(image_src_straight, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 9, 2)
        #image_src_straight = np.invert(image_src_straight)

        #cv2.imshow("H", image_src_straight)
        #cv2.imshow("Canny", image_src)
        
        print e1 - s1, ".", e2 - s2
        return top_view


def get_median(l):
        global Q
        Q.pop()
        Q.appendleft(l)
        med = []
        for elem in Q:
                med.append(elem)			
        return np.median(med)

def trs(image_src):
	lines = 0
	#			image, -, -, threshold, maxLineGap, minLineLenght, 
	lines = cv2.HoughLinesP(image_src,1, np.pi/2, 10, 2, 25) # 2, 60)
	if lines is not None:
                for line in lines:
                        try:
				coords = line[0]
        #                        cv2.line(image_src, (coords[0], coords[1]), (coords[2], coords[3]), [100], 3)
                        except:
                                pass
        return 0

def image_display(image_src):#l
        #font = cv2.FONT_HERSHEY_SIMPLEX
	#cv2.putText(image_src, str(l), (3,30), font, 0.7, (120), 2, 0)
	#if lines is None:
	#	cv2.putText(image_src, 'Curve', (3,30), font, 0.5, (255,255,255), 2, 0)
			
	#else:
	#	cv2.putText(image_src, str(l) , (3,30), font, 0.5, (255,255,255), 2, 0)
	
	if image_src is not None:
                #image_src = imutils.resize(image_src, width=400)
		#cv2.namedWindow('image_src', WINDOW_NORMAL)
	#	cv2.imshow('image_src', image_src)
		pass
		
	#	cv2.moveWindow('image_src', 700, 700)

def undistort(img):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def init():
    cv2.namedWindow("slider")
    cv2.createTrackbar("HUE_top", "slider", 0, 180, nothing)
    cv2.createTrackbar("SATURATION_top", "slider", 0, 255, nothing)
    cv2.createTrackbar("VALUE_top", "slider", 0, 255, nothing)
    
    cv2.createTrackbar("HUE_bot", "slider", 0, 180, nothing)
    cv2.createTrackbar("SATURATION_bot", "slider", 0, 255, nothing)
    cv2.createTrackbar("VALUE_bot", "slider", 0, 255, nothing)
    
    cv2.createTrackbar("bot", "slider", 0, 255, nothing)
    cv2.createTrackbar("top", "slider", 0, 255, nothing)

def histogram(top_view):
        #Bild ist 140 hoch x 120 breit pixel
        s1 = top_view[130:140,:]
        #s1= cv2.resize(top_view[130:140,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        
        #s2 = top_view[120:130,:]
        s2= cv2.resize(top_view[120:130,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        
        #s3 = top_view[110:120,:]
        s3= cv2.resize(top_view[110:120,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        
        #s4 = top_view[100:110,:]
        s4= cv2.resize(top_view[100:110,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        
        #s5 = top_view[90:100,:]
        s5= cv2.resize(top_view[90:100,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        
        #s6 = top_view[80:90,:]
        s6= cv2.resize(top_view[80:90,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        
        #s7 = top_view[70:80,:]
        s7= cv2.resize(top_view[70:80,:], None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        

        
        s1_plot = np.zeros((10,120,1), np.uint8)
        
        s1_list = []
        #for col in range(10):
        #   s1_list.append(cv2.countNonZero(s1[col,:]))
        for col in range(120):
            temp = cv2.countNonZero(s1[:,col])
            s1_list.append(temp)  #^,->      links Zeile, rechts Spalte
            #s1_plot[col, temp] = [255]
            cv2.circle(s1_plot,(col, temp), 1, (170), -1)
            
            
        cv2.imshow("1", cv2.resize(s1, None, fx = 6, fy = 6, interpolation = cv2.INTER_CUBIC))
        cv2.imshow("1_plot)", cv2.resize(s1_plot, None, fx = 6, fy = 6, interpolation = cv2.INTER_CUBIC))
        #cv2.imshow("2", s2)
        #cv2.imshow("3", s3)
        #cv2.imshow("4", s4)
        #cv2.imshow("5", s5)
        #cv2.imshow("6", s6)
        #cv2.imshow("7", s7)       
        
        
            
        return 0

if __name__ == "__main__":
        start_new_thread(get_acc_data, (None,))
        #start_new_thread(gyro_regelung, (None,))
        print("[INFO] Init Cam")
        time.sleep(1.0)
        fps = FPS().start()
        for (i, f) in enumerate(stream):
                # grab the frame from the stream and resize it to have a maximum
                # width of 400 pixels
                frame = f.array
                #frame = imutils.resize(frame, width=400)
         
                # check to see if the frame should be displayed to our screen
                
#                cv2.imshow("Frame", frame)
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
	init()
	if RUNNING_ON_PI == True:
            
            while True:
            #cap = cv2.VideoCapture('2018_06_20_2.h264')
            #	frame_counter = 0	
                init()
								#Dieser Block wird ausgefuehrt wenn das Programm von der Kamera Video beziehen soll
		
		#for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		while True:        
		        start1 = time.time()
		        #image_src = frame.array
                        image_src = vs.read()
                        end1 = time.time()
		        start2 = time.time()
		        top_view = image_proc(image_src)
		        lane_maximums = histogram(top_view)
		        end2 = time.time()
		        
		        start3 = time.time()
		        #l = trs(image_src)
		        end3 = time.time()
		        #out.write(image_src)

		        start4 = time.time()
		        #image_display(image_src) #cap,l
		        end4 = time.time()
		        #rawCapture.truncate(0)      
		        #bla = (end2-start2)*1000+(end3-start3)*1000
		        #print (end1 - start1)*1000, "," , (end2-start2)*1000, "," , (end3-start3)*1000
		        
		        if cv2.waitKey(1) & 0xFF == ord('q'):
                                set_motor_dutycycle(0)
		                #cap.release()
	    #			cv2.destroyAllWindows()
                                stream.stop()
		                break
		        fps.update()
		        
	else:					#Dieser Block wird ausgefuehrt, wenn das Programm von einer Datei das Video lesen soll
                cap = cv2.VideoCapture('2018_08_02_2.h264')
		while True:
			start1 = time.time()		
			ret, image_src = cap.read()
			end1 = time.time()
			
			#image_src = undistort(image_src)
			
			start2 = time.time()
			top_view = image_proc(image_src)
			lane_maximums = histogram(top_view)
			end2 = time.time()
			
			start3 = time.time()
			#l = trs(image_src)
			end3 = time.time()

			start4 = time.time()
			#image_display(image_src) #cap,l
			end4 = time.time()     
	
			#bla = (start1-end1)*1000+(end2-start2)*1000+(end3-start3)*1000
			#perf.append(bla)
			#print np.median(perf)
			if cv2.waitKey(1) & 0xFF == ord('q'):
			        set_motor_dutycycle(0)
			        #cap.release()
	#			cv2.destroyAllWindows()
			        break 	

