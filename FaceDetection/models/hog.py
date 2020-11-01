# -*- coding: utf-8 -*-"""Created on Fri Oct 30 14:46:44 2020@author: Ayman Al Jabri"""import dlibimport cv2kwargs = {        'scaleFactor':1.1,        'minNeighbors':4,        'minSize':(50,50),        'maxSize':None        }fname = 'haarcascade_frontalface_default.xml'class HAAR(dlib.get_frontal_face_detector):    def __init__(self):        super(HAAR, self).__init__()    def find_faces(self,img):        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        return self.detectMultiScale(gray, **self.kwargs)    def draw_rect(self, frame, faces):        if faces  is None: return frame        for box in faces:            x,y,h,w = box            cv2.rectangle(frame, (x, y), (x+h, y+w), (80,18,236), 2)            cv2.rectangle(frame, (x, y), (x+h, y-15), (80,18,236), cv2.FILLED)            cv2.putText(frame, 'face', (x + 6, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)        return frame