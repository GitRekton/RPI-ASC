import cv2
import numpy as np
import time
import math
from collections import deque
lines = None


#FLAGS
RUNNING_ON_PI = True	#Abhaengig von der aktuellen Laufzeitumgebung
MOTOR_ACTIVE = False	#Debugvariable- aktiviert den Motor, waehrend Debug ausgeschaltet
BEEPER_ACTIVE = False	#Debugvariable- aktiviert den Beeper, waehrend Debug ausgeschaltet
THREAD_STARTED = False	#Flag fuer Mutlithreading

Q = deque(4*[0], 4)		#Groesse des Schieberegisters, welches als Filter fungiert. Mittelwert des Registers = aktuelelr abstand zur naechsten Kurve
print "Init Queue"


def absolute(x):
	return -x if x < 0 else x

def nothing(x):
	pass

def image_proc(image_src):
    #Image Colorspace
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)	#LaneTracking
        #image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV) #LaneSideTracking with ColorFilter

    #Image Crop
        #320,240
        #print image_src.shape[0]
        #					  Zeilen, 	Spalten
        #		[Horizont:Motorhaube, links:rechts]
        image_src = image_src[image_src.shape[0] * 0.41:image_src.shape[0], image_src.shape[1] * 0.4:image_src.shape[1] * 0.59]	# [200:400, 250:300]	
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
			#print l_med
			pass
		elif l_med < 180:
			pass
			#print l_med
                if RUNNING_ON_PI == False:
                    cv2.line(image_src, (int(med_x1), int(med_x2)), (int(med_y1), int(med_y2)), [160], 20)
                    #cv2.line(image_src, (coords[0], 240), (coords[0], 240 - int(l_med)), [100], 20)
                    #cv2.line(image_src, (0, 240 - int(l_med)),(250,240 - int(l_med)), [100], 9) 
	else:
		pass

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
        

if __name__ == "__main__":
        print("[INFO] Init Cam")
        time.sleep(2.0)
	perf = []
	if RUNNING_ON_PI == True:		        
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
	
			bla = (end1 - start1)*1000+(end2-start2)*1000+(end3-start3)*1000
			perf.append(bla)
			print (end1 - start1)*1000,(end2-start2)*1000, (end3-start3)*1000
			#print np.median(perf)
			if cv2.waitKey(1) & 0xFF == ord('q'):
			        set_motor_dutycycle(0)
			        #cap.release()
	#			cv2.destroyAllWindows()
			        break 	
