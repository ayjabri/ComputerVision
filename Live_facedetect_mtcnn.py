# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:44 2020

@author: Ayman Al Jabri
This script will detect faces via your webcam using multithread
There should be no delay as a result of getting the faces from the model.
Tested with OpenCV
"""

import cv2
import queue
import threading
import argparse
from FaceDetection.models import f_net, haar




class FaceDetectLive(object):
    def __init__(self, classifier, skip_n=1, h_res=400, v_res=600, font=cv2.FONT_HERSHEY_DUPLEX, th = False):
        self.clf = classifier
        self.skip_n = skip_n
        self.h_res = h_res
        self.v_res = v_res
        self.font = font
        self.VideoCapture()
        self.q = queue.deque(maxlen=100)
        self.th = th

    def VideoCapture(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.h_res)
        self.cap.set(4, self.v_res)

    def timer(self, i):
        return (i % self.skip_n == 0)
    
    def thread(self, frame):
        th = threading.Thread(target=self.find_faces, args=(frame,))
        th.start()
    
    def find_faces(self, frame):
        faces = self.clf.find_faces(frame)
        self.q.append(faces)
        
    def play(self):
        i = 0
        faces = None
        while True:
            good, frame = self.cap.read()
            frame = cv2.flip(frame,1) #flip horizontaly because it looks better!
            if good:
                if self.timer(i):
                    if self.th:
                        self.thread(frame)
                        faces = self.q.pop() if len(self.q) > 0 else None
                    else:
                        faces = self.clf.find_faces(frame)
                        
                frame = self.clf.draw_rect(frame, faces)
                i += 1
                cv2.imshow('face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):break
        self.cap.release()
        cv2.destroyAllWindows()
        self.q.clear()
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=False, help='Name and path of the HAAR file')
    parser.add_argument('--n', required=False, type= int, default=1, help='Detect faces on the Nth frame')
    parser.add_argument('--algo', required=False, help='Algorithm to use: choose between: "haar", "hog" and "facenet"')
    parser.add_argument('--threading', required=False, default=False, type=str, help='Use threading to detect faces')
    arg = parser.parse_args()
    fname = (arg.file if arg.file else 'FaceDetection/models/haarcascade_frontalface_default.xml')

    if arg.algo == 'facenet':
        clf = f_net.FaceNet(**f_net.params)
    else:
        clf = haar.HAAR(fname)


    cam = FaceDetectLive(clf, skip_n=arg.n, th=arg.threading)
    cam.play()

