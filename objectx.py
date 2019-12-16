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

from core import utils
import json
import numpy as np
import datetime
import os
import base64
import requests
import ast

import multiprocessing as mp

from argsdk import imgsize, batchsize
import cv2
import time
import numpy as np

from flask import Flask, request, jsonify
flaskserver = Flask(__name__)


global detection

# Handle API Call
@flaskserver.route(rule='/detect', methods=['POST'])
def detect():

    global detection
    content = request
    # batch = np.fromstring(content.data, np.float32).reshape(
    #     (-1,imgsize, imgsize, 3))

    data = ast.literal_eval(content.data.decode('utf-8'))
    # print(data['height'],data['width'])

    batch = np.fromstring(data['Batch'], np.uint8).reshape((data['height'], data['width'], 3))
    # print("enter")
    # print('############img shape')
    # print(batch.shape)
    # print('#####################')
    # cv2.imshow("image", batch)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    inferenceList = detection.run(batch)
    return jsonify({"Response": str(inferenceList)})
    # decoded_string = base64.decodebytes(result)
    # image = cv2.imdecode(np.fromstring(decoded_string, np.uint8), 1)

class Detection():
    # cfgfile = "./yolov3.cfg"
    # weightsfile = "/home/dinesh/darknet-master/backup4000ocr/yolov3_8000.weights"
    # classes = load_classes('./obj.names')
    # num_classes = 36
    # bbox_attrs = 5 + num_classes
    # args = None
    # confidence = 0.1
    # nms_thesh = 0.1
    # # CUDA = False
    # # model = None
    # # inp_dimensions = None
    # colors = list()

    # def __init__(self):
    #     ''' Called when class object is created. '''

    #     self.img_size = imgsize
    #     self.max_batch_size = batchsize
    #     self.num_classes = 36

    #     self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(), "./yolov3_cpu_nms.pb",
    #                                                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])
    #     # self.input_tensor_p, self.output_tensors_p = utils.read_pb_return_tensors(tf.get_default_graph(), "/home/injamurikrutika/Desktop/darknet-master/machinepb/yolov3_cpu_nms.pb",
    #     #                                                                          ["Placeholder:0", "concat_9:0", "mul_6:0"])
    #     self.config = tf.ConfigProto()
    #     self.config.gpu_options.per_process_gpu_memory_fraction = 0.85
    #     self.sess = tf.Session(config=self.config)
    #     _ = self.sess.run(self.output_tensors, feed_dict={
    #         self.input_tensor: self.createRandomSample()})

    def color_generator(self):
        ''' Generate a color pallete for different objects.
        '''

        for i in range(0, 36):
            temp = list()
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            temp.append(b)
            temp.append(g)
            temp.append(r)
            self.colors.append(temp)

    def plot_results(self, output, img):
        ''' 
        Draw the bounding box and results on the frame.
        '''
        x1 = []
        y1 = []
        x2  = []
        y2 = []
        score = []
        labels = []
        height = 0
        for x in output:

            if float(x[5]) <= 0 or float(x[5]) > 1:
                continue

            cls = int(x[-1])
            if int(cls)<0 or int(cls)>36:
                continue

            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            xx1 = c1[0].item()
            yy1 = c1[1].item()
            xx2 = c2[0].item()
            yy2 = c2[1].item()
            x1.append(c1[0].item()) 
            y1.append(c1[1].item())
            x2.append(c2[0].item())
            y2.append(c2[1].item())
            if height == 0:
                height = c2[1].item() - c1[1].item()
                print(height)
            label = "{0}".format(self.classes[cls])
            scores = str("{0:.3f}".format(float(x[5])))
            labels.append("{0}".format(self.classes[cls]))
            score.append(str("{0:.3f}".format(float(x[5]))))
            color = self.colors[cls]
            
            cv2.rectangle(img, (xx1, yy1), (xx2, yy2), color, 2)
            cv2.rectangle(img, (xx1, yy1), (xx1 + (len(label) + len(scores)) * 10, 
                            yy1 - 10) , color, -1, cv2.LINE_AA)
            cv2.putText(img, label + ':' + scores, (xx1, yy1), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return x1,y1,x2,y2, score, labels, height, img
        #return img

    # def argsparser(self):
    #     '''
    #     Argument parser for command line arguments.
    #     '''

    #     parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    #     parser.add_argument('--confidence', dest='confidence', help='Object Confidence to filter predictions', default=0.25, type=float)
    #     parser.add_argument('--nms_thresh', dest='nms_thresh', help='NMS Threshhold', default=0.1, type=float)
    #     parser.add_argument('--reso', dest='reso', help=
    #                         'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',
    #                         default=416, type=int)
    #     parser.add_argument('--source', dest='source', default=0, help='Input video source', type=str)
    #     parser.add_argument('--skip', dest='skip', default=1, help='Frame skip to increase speed', type=int)
    #     return parser.parse_args()
    
    def run(self,batch):
        '''
        Method to run the detection.
        '''


        # cap = cv2.VideoCapture(self.args.source)
        # assert cap.isOpened(), 'Cannot capture source'

        #frame = cv2.imread("./c.jpg")

        # path = "./img/"
        start = time.time()  

        # while cap.isOpened():
        
        #     ret, frame = cap.read()
        #     if ret:
        #         if not frames % self.args.skip == 0:
        #             frames += 1
        #             continue
        # for framex in os.listdir(path):
        # frame = cv2.imread(path+framex)
        frame = batch
        framey = batch.copy()
        blank_image = np.zeros((300,300,3), np.uint8)
        img, orig_im, dim = prep_image(frame, self.inp_dimensions)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)                        
    
        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
    
        output = self.model(Variable(img), self.CUDA)
        # print('#################output of network')
        # print(output)
        # print('##################################')
        output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thresh)

        # print('#################output after nms')
        # print(output)
        # print('##################################')
        if not type(output) == int:

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(self.inp_dimensions))/self.inp_dimensions

            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]
    
            x1, y1, x2, y2, scores, labels, height, orig_im = self.plot_results(output,orig_im)
            chars = plate.coordinates(x1, y1, x2, y2, height, scores, labels, self.classes) 
        print("output",chars)
        
        try:
            cv2.putText(blank_image,"{}".format(','.join(chars)),(2,100),cv2.FONT_HERSHEY_COMPLEX,1,[0,0,255],1)
        except:
            print("")
        framey = cv2.resize(framey,(300,300))
        imstack = np.concatenate((framey, blank_image), axis=1)
        if chars is None:
            return ([])
        else:
            return ((chars))
            # cv2.imwrite("/home/dinesh/YOLOv3-PyTorch-master/imgs/{}.jpg".format(framex),imstack)
            # cv2.imshow(self.windowName, orig_im)
            # # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        #frames += 1
                #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)), end='\r')
    
    def __init__(self):
        '''
        Intitialize method to run on class object creation.
        '''
        
        # self.args = self.argsparser()
        self.cfgfile = "./yolov3.cfg"
        self.weightsfile = "/home/dinesh/darknet-master/backup4000ocr/yolov3_8000.weights"
        self.classes = load_classes('./obj.names')
        self.num_classes = 36
        self.bbox_attrs = 5 + self.num_classes
        # args = None
        # confidence = 0.1
        # nms_thesh = 0.1
        self.CUDA = False
        self.model = None
        self.inp_dimensions = None
        self.colors = list()

        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            print('Device Used: ', torch.cuda.get_device_name(0))
            print('Capability: ', torch.cuda.get_device_capability(0))
        self.confidence = 0.01
        self.nms_thresh = 0.1
        self.reso = 416
        # self.confidence = float(self.confidence)
        # self.nms_thesh = float(self.nms_thresh)

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)
        self.model.net_info["height"] = self.reso

        self.inp_dimensions = int(self.model.net_info["height"])

        assert self.inp_dimensions % 32 == 0, 'Input not a multiple of 32'
        assert self.inp_dimensions > 32, 'Input must be larger than 32'


        if self.CUDA:
            self.model.cuda()

        self.model.eval()

        self.color_generator()
        # self.windowName = "Object Detection"
        # cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# if __name__ == '__main__':

#     detection = Detection()
#     detection.run()

global detection
detection = Detection()


if __name__ == '__main__':
    flaskserver.run(host='127.0.0.1',
                    port=5000,
                    debug=True)    
