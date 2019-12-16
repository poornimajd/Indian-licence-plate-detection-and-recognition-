
from utils import normalize
import time
import torch
from torch.autograd import Variable
import cv2
from scripts.util import *
from scripts.darknet import Darknet
from scripts.preprocess import prep_image
import argparse
import random
import plate
import os
import requests
import ast

path = "/home/dinesh/Desktop/sarvesh/ocr/img/"
for i in os.listdir(path):
    frame = cv2.imread(path+i)
    # print(frame.shape)

    data = {}
    data['Batch'] = frame.tostring()
    data['height'] = frame.shape[0]
    data['width'] = frame.shape[1]
    response = requests.post('http://127.0.0.1:8000/detect', data=str(data))

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # # n_frame, frame = normalize(frame, 416)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # response = requests.post(
    #     'http://127.0.0.1:8000/detect', data=frame.tostring())

    result = ast.literal_eval(ast.literal_eval(
        response.content.decode('utf-8'))['Response'])
    # print(result)
    print("".join(result))
        
# image = cv2.imread(path)
# height, width, channels = image.shape[0], image.shape[1], image.shape[2]
# image = cv2.imencode('.jpg', image)[1]
# encoded_string = base64.b64encode(image)
# inference_endpoint = 'http://127.0.0.1:8000/detect'
# response = requests.post(inference_endpoint, data=encoded_string)
# result = ast.literal_eval(ast.literal_eval(response.content.decode('utf-8'))['Response'])
# print(result)