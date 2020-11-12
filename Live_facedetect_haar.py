'''
This script will detect faces via your webcam using multithread
There should be no delay as a result of getting the faces from the model.
Tested with OpenCV
'''
import cv2
import queue
import threading


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.open(0)


# Create the haar cascade
faceCascade = cv2.CascadeClassifier("FaceDetection/models/haarcascade_frontalface_default.xml")
tracker = cv2.TrackerCSRT_create()


idx = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if not ret:
        break
    idx += 1
    if idx % 30 == 0:
        box = faceCascade.detectMultiScale(frame,minNeighbors=5,minSize=(30, 30))
        tracker.init(frame, tuple(box[0]))
    try:
        success, tbox = tracker.update(frame)
        tbox = [int(x) for x in tbox]

        (x, y, w, h) = tbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    except:
        pass

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()