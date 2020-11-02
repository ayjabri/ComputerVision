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
        'image_size':160,
        "margin":0,
        "min_face_size":50,
        "thresholds":[0.6, 0.7, 0.7],
        "factor":0.709,
        "post_process":True,
        "select_largest":True,
        "selection_method":"probability",
        "keep_all":True,
        "device":None
        }

class FaceNet(facenet.MTCNN):
    def __init__(self, **kwargs):
        super(FaceNet, self).__init__(**kwargs)


    def find_faces(self,img):
        return self.detect(img)[0]

    def draw_rect(self, frame, faces, text='Searching...'):
        if faces  is None: return frame
        img = frame.copy()
        for box in faces:
            x,y,h,w = box.astype(int)
            cv2.rectangle(img, (x, y), (h, w), (80,18,236), 2)
            cv2.rectangle(img, (x, y), (h, y-15), (80,18,236), cv2.FILLED)
            cv2.putText(img, text, (x + 6, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        return img

    def crop(self, img, box, scale=[75,30]):
        tr,tl,lr,ll = box.astype(int)
        tl -=scale[0]
        tr +=scale[0]
        lr +=scale[1]
        ll -=scale[1]
        return img[tl:tr,ll:lr]

net = facenet.InceptionResnetV1(pretrained='vggface2').eval()
