# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:46:44 2020

@author: Ayman Al Jabri
"""

import facenet_pytorch as facenet
import cv2
# try:
#     import facenet_pytorch as facenet
# except Error:
#     print('''Please install facenet_pytorch 
#     run: $pip install facenet_pytorch''')
#     pass


params = {
        'image_size':260,
        "margin":2,
        "min_face_size":20,
        "thresholds":[0.6, 0.7, 0.7],
        "factor":0.709,
        "post_process":True,
        "select_largest":True,
        "selection_method":"largest_over_theshold",
        "keep_all":True,
        "device":None
        }

class FaceNet(facenet.MTCNN):
    def __init__(self, **kwargs):
        super(FaceNet, self).__init__(**kwargs)
        
    
    def find_faces(self,img):
        return self.detect(img)[0]
    
    def draw_rect(self, frame, faces):
        if faces  is None: return frame
        for box in faces:
            x,y,h,w = box.astype(int)
            cv2.rectangle(frame, (x, y), (h, w), (80,18,236), 2)
            cv2.rectangle(frame, (x, y), (h, y-15), (80,18,236), cv2.FILLED)
            cv2.putText(frame, 'face', (x + 6, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        return frame #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

