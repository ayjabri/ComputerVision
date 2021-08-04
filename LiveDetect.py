# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:44 2020

@author: Ayman Al Jabri
This script multithreads the detection of faces on webcam 
There should be no delay as a result of getting the faces from the model.
Tested with OpenCV
"""

import cv2
import queue
import joblib
import threading
import argparse
from FaceDetection.models import f_net, haar, hog
import pandas as pd

classes = pd.read_csv('FaceRecognition/results/classes.csv')
classes = classes.values.reshape(len(classes))

class FaceDetectLive(object):
    '''
    Create a face-detection object using webcam.
    classifier: choices between "HAAR", "FaceNet" or "HOG"
    recognize: path to an sklearn modle trained on features extracted from Facenet
    '''
    def __init__(self, classifier, recognize=None, skip_n=1,
                 h_res=400, v_res=600, font=cv2.FONT_HERSHEY_DUPLEX, th = False):
        self.clf = classifier
        self.recognize = recognize
        if recognize is not None:
            self.model = joblib.load(recognize)
            self.net = f_net.net
        self.skip_n = skip_n
        self.h_res = h_res
        self.v_res = v_res
        self.font = font
        self.__VideoCapture__()
        self.q = queue.deque(maxlen=100)
        self.th = th
        self.idx = 0

    def __VideoCapture__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.open(0)
        self.cap.set(3, self.h_res)
        self.cap.set(4, self.v_res)


    def __timer__(self):
        return (self.idx % self.skip_n == 0)

    def __thread__(self, frame):
        th = threading.Thread(target=self.__append_faces__, args=(frame,))
        th.start()

    def __append_faces__(self, frame):
        faces = self.clf.find_faces(frame)
        self.q.append(faces)

    def play(self):
        faces_old,old_names,faces = None, None, None
        while True:
            good , frame = self.cap.read()
            if not good:
                continue
            frame = cv2.flip(frame,1) #flip horizontaly because it looks better!
            if self.__timer__():
                if self.th:
                    self.__thread__(frame)
                else:
                    self.__append_faces__(frame)
                faces = self.q.pop() if len(self.q) > 0 else faces_old
                if faces is not None and self.recognize and self.idx % 30 ==0 :
                    try:
                        ii = self.clf(frame)
                        features = self.net(ii).detach()
                        d = self.model.predict(features).tolist()
                        names = classes[d]
                        old_names = names
                    except:
                        pass
                faces_old = faces
            frame = self.clf.draw_rect(frame, faces_old, old_names)
            self.idx += 1
            cv2.imshow('face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):break
        self.cap.release()
        cv2.destroyAllWindows()
        self.q.clear()
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        help='Name and path of the HAAR file')
    parser.add_argument('--n', type= int, default=1,
                        help='Detect faces on the Nth frame')
    parser.add_argument('--algo',
                        help='Algorithm to use: choose between: "haar", "hog" and "facenet"')
    parser.add_argument('--threading', default=False, action='store_true',
                        help='Use threading to detect faces')
    parser.add_argument('--recognize', action='store_true', default =False,
                        help='recognize the face using features')

    arg = parser.parse_args()
    pathM = None
    fname = (arg.file if arg.file else 'FaceDetection/models/haarcascade_frontalface_default.xml')
    if arg.algo == 'facenet' or arg.recognize:
        clf = f_net.FaceNet(**f_net.params)
        if arg.recognize:
            pathM = 'FaceRecognition/model.joblib'
    elif arg.algo =='hog':
        clf = hog.HOG()
    else:
        clf = haar.HAAR(fname)


    cam = FaceDetectLive(clf, recognize= pathM, skip_n=arg.n, th=arg.threading)
    cam.play()