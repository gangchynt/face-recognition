import os
import sys
import imutils
import numpy as np
import cv2
from base64 import b64decode
import argparse
import datetime

fileDir = os.path.dirname(os.path.realpath(__file__))
# print(fileDir)
modelDir = os.path.join(fileDir, 'models')


parser = argparse.ArgumentParser()
parser.add_argument('--faceDetectorModel', type=str, help="Path to Face detection model",
                    default=os.path.join(modelDir, "res10_300x300_ssd_iter_140000.caffemodel"))
parser.add_argument('--prototextPath', type=str, help="Path to prototxtPath",
                    default=os.path.join(modelDir, "deploy.prototxt"))
parser.add_argument('--faceRecognitionModel', type=str, help="Path to Face recognition model",
                    default=os.path.join(modelDir, "nn4.small2.v1.t7"))
parser.add_argument("--txt", help="VGG's directory of text files of people with images.",
                    default='raw-txt')
parser.add_argument("--raw", help="Directory to save raw images to.",
                    default='raw')
parser.add_argument("--aligned", help="Directory to save aligned images to.",
                    default='aligned')

args = parser.parse_args()

faceDetectorModel = args.faceDetectorModel
prototextPath = args.prototextPath
faceRecognitionModel = args.faceRecognitionModel

# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []

#Load Model
print("Loading model...................")
#load face detection model
detectionNet = cv2.dnn.readNetFromCaffe(prototextPath,faceDetectorModel)
detectionNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
detectionNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
recognitionNet= cv2.dnn.readNetFromTorch(faceRecognitionModel)

stream = cv2.VideoCapture("rtsp://admin:admin@192.168.200.102:554/1")
while(1):
    ret, frame =stream.read()
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #accessing the image.shape tuple and taking the elements
    (h, w) = rgbImage.shape[:2]

    #get our blob which is our input image 
    # For example, the mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68 
    # (you may have already encountered these values before 
    # if you have used a network that was pre-trained on  ImageNet).
    # ref:https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(cv2.resize(rgbImage, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # apply OpenCV's deep learning-based face detector to localize
	  # faces in the input image
    detectionNet.setInput(blob)
    detections = detectionNet.forward()

    #Iterate over all of the faces detected and extract their start and end points
    count = 0
    for i in range(0,detections.shape[2]):
      box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")
      confidence = detections[0, 0, i, 2]
      #if the algorithm is more than 16.5% confident that the detection is a face, show a rectangle around it
      if (confidence > 0.165):
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        count = count + 1
        face= frame[startY:endY,startX:endX]
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				  (96, 96), (0, 0, 0), swapRB=True, crop=False)
        recognitionNet.setInput(faceBlob)
        vec = recognitionNet.forward()
        knownNames.append(i)
        knownEmbeddings.append(vec.flatten())
        print("[INFO] serializing {} encodings...".format(range(0,detections.shape[2])))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(args["embeddings"], "wb")
        f.write(pickle.dumps(data))
        f.close()

    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)

    
# def processFrame(frame)