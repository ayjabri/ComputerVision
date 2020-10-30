# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:44 2020

@author: Ayman Al Jabri
"""

import cv2
import argparse
import matplotlib.pyplot as plt
from models import facenet, haar



class FaceDetectLive(object):
    def __init__(self, classifier, skip_n=1, h_res=800, v_res=600, font=cv2.FONT_HERSHEY_DUPLEX):      
        self.clf = classifier
        self.skip_n = skip_n
        self.h_res = h_res
        self.v_res = v_res
        self.font = font
        self.VideoCapture()
    
    def VideoCapture(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.h_res)
        self.cap.set(4, self.v_res)

    def timer(self, i):
        return (i % self.skip_n == 0)

    def detect(self):
        i = 0
        faces = None
        while True:
            good, frame = self.cap.read()
            frame = cv2.flip(frame,1) #flip horizontaly because it looks better
            if good:
                if self.timer(i) :
                    faces = self.clf.find_faces(frame)
                frame = self.clf.draw_rect(frame, faces)
                i += 1
            cv2.imshow('face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):break
        self.cap.release()
        cv2.destroyAllWindows()
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=False, help='Name and path of the HAAR file')
    parser.add_argument('--n', required=False, type= int, default=1, help='Detect faces on the Nth frame')
    parser.add_argument('--algo', required=False, help='Algorithm to use: choose between: "haar", "hog" and "facenet"')
    arg = parser.parse_args()
    fname = (arg.file if arg.file else 'models/haarcascade_frontalface_default.xml')

    if arg.algo == 'facenet':
        from models import facenet
        clf = facenet.FaceNet(**facenet.params)
    else:
        from models import haar
        clf = haar.HAAR(fname)
    
    
    cam = FaceDetectLive(clf, arg.n)
    cam.detect()

    