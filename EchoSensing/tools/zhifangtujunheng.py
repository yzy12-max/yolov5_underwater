import cv2
import numpy as np

imgpath = "/home/yzy/datasets/yolo_voice/val.txt"

def enhanceImage(imgpath):
    img = cv2.imread(imgpath)
    img1 = cv2.split()



