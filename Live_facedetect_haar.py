'''
This script will detect faces via your webcam using multithread
There should be no delay as a result of getting the faces from the model.
Tested with OpenCV
'''
import cv2
import queue
import threading


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
q = queue.deque()

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect(frame):
    q.append(faceCascade.detectMultiScale(frame,
                                          scaleFactor=1.4,
                                    	  minNeighbors=5,
                                          minSize=(30, 30)))
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th1 = threading.Thread(target= detect, args= (gray,))
        th1.start()
        try:
            faces = q.pop()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        except:
            pass
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




