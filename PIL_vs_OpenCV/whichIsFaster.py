'''
This small test is to find out which library is faster in prcessing photos in python: PIL SIMD or OpenCV
Summary:
On resize: PIL is 2.3 times faster in resizing a photo
grayscaling: OpenCV is faster when it comes ot grayscaling a photo

Conclusion: if you want to grayscale then resize a photo for DeepLearning then you are better off using OpenCV

'''

from timeit import timeit
import cv2
import argparse


cv2_resize = '''
resized = cv2.resize(img, (260, 260), False)
'''

cv2_resize_gray = '''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (260, 260), False)
'''

cv2_setup = '''
import cv2
img = cv2.imread("img.jpg")
'''

PIL_resize = '''
# To make the two copmarable we need to resize the photo first them convert it to numpy array
resized = np.array(img.resize((260,260), resample=Image.NEAREST, reducing_gap=3))
'''

PIL_resize_gray = '''

resized = img.resize((260,260), resample=Image.NEAREST, reducing_gap=3)
gray = np.array(resized.convert("L"))
'''

PIL_setup =  '''
from PIL import Image
import numpy as np
img = Image.open('img.jpg')
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gray',action='store_true', default=False,help='Grayscale the photo before resizing it')
    parser.add_argument('--n', required=False, default= 100, type=int, help='Number of times to run the codes')
    arg = parser.parse_args()
    if arg.gray:
        cv2_resize = cv2_resize_gray
        PIL_resize = PIL_resize_gray
    
    cv = timeit(stmt=cv2_resize, setup=cv2_setup, number=arg.n)
    PIL = timeit(stmt=PIL_resize, setup=PIL_setup, number=arg.n)

    state = ('faster' if cv/PIL > 1 else 'slower')
    print(f'PIL is {cv/PIL:.1f} times {state} that OpenCV')