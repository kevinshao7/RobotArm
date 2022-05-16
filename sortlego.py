#Kevin Shao 2022

import time
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from datetime import datetime
import arducam_mipicamera as arducam
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.point_cloud import PointCloud
from stereovision.stereo_cameras import CalibratedPair
from stereovision import *
import pandas as pd
import netCDF4
from PIL import Image


def align_down(size, align):
    return (size & ~((align)-1))

def align_up(size, align):
    return align_down(size + align - 1, align)

def get_frame(camera):
    frame = camera.capture(encoding = 'i420')
    fmt = camera.get_format()
    height = int(align_up(fmt['height'], 16))
    width = int(align_up(fmt['width'], 32))
    image = frame.as_array.reshape(int(height * 1.5), width)
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    image = image[:fmt['height'], :fmt['width']]
    return image

camera_params = json.load(open("camera_params.txt", "r"))
camera = arducam.mipi_camera()
print("Open camera...")
camera.init_camera()
mode = camera_params['mode']
camera.set_mode(mode)
fmt = camera.get_format()
height = int(align_up(fmt['height'], 16))
width = int(align_up(fmt['width'], 32))
print("Current mode: {},resolution: {}x{}".format(fmt['mode'], fmt['width'], fmt['height']))
print("Enable Auto Exposure...")
camera.software_auto_exposure(enable = True)
print("Enable Auto White Balance...")
camera.software_auto_white_balance(enable = True)
# Camera settimgs
cam_width = fmt['width']
cam_height = fmt['height']

img_width= camera_params['width']
img_height = camera_params['height']
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))


# camera.set_control(0x00980911, 1000)
# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')



disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)
CalibratedPair = CalibratedPair(None, calibration, sbm)

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    #local_max = -510
    #local_min = -1504
    local_max = disparity.max()
    local_min = disparity.min()
    print("maxmin",local_max,local_min)
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    cv2.rectangle(disparity_color,[346, 117,  53,  24],color=(255,0,0),thickness=2)
    cv2.imshow("Image", disparity_color)
    key = cv2.waitKey(1) & 0xFF 
    return disparity

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Parameters loaded from file '+fName)


# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)


load_map_settings ("3dmap_set.txt")

img = cv2.imread('Lenna.png')
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
#camera.software_auto_white_balance(enable = True)

img = Image.open('300.png')
rgb_img = img.convert('RGB')
rgb_img.save('300.jpg')

classNames = []
classFile='obj.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
configPath = 'yolov4-custom.cfg'
weightsPath = 'yolov4-custom_7000.weights'

net= cv2.dnn.readNetFromDarknet(configPath,weightsPath)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)


while True:

    # show the frame   
    frame = get_frame(camera)
    frame = cv2.resize(frame, (img_width, img_height))

    img = frame [0:img_height,0:int(img_width/2)] #Left image Y+H and X+W
    imgL = img #Left image Y+H and X+W
    imgR = frame [0:img_height,int(img_width/2):img_width] #Left image Y+H and X+W
    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    disparity = stereo_depth_map(rectified_pair)
    # show the frame   
    #print(classIds,bbox)
    t2 = datetime.now()
    #print ("DM build time: " + str(t2-t1))
    classIds,confs,bbox=model.detect(img,confThreshold=0.9 )
    idxs = cv2.dnn.NMSBoxes(bbox, confs, 0.9,
	0.9)
    print(idxs) 
    
    if len(idxs) != 0:
        for i in idxs:
                box = bbox[i]
                confidence = confs[i]
                classId = classIds[i]
                #print(box)
                #targetbox = img[int((box[1]+(box[0]+box[2])/2)):int(box[1]+(box[0]+box[2])/2+1),int((box[0]+box[2])/2):int((box[0]+box[2])/2+1)]
                #print(targetbox)
                startpoint = (int(box[0]+box[2]/4),int(box[1]+box[3]/2))
                endpoint = (int(box[0]+box[2]/4)+10,int(box[1]+box[3]/2)+10)
                boxa = (box[0],box[1])
                boxb = (box[0]+box[2],box[1]+box[3])
                #print(startpoint)
                #print(endpoint)
                cv2.rectangle(img,box,color=(255,0,0),thickness=2)
                cv2.putText(img,classNames[classId].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0 ,0),2)
                #cv2.rectangle(img,startpoint,endpoint,color=(255,255,0),thickness=2)
                width = box[2]
                height = box[3]
                boxpoints = np.zeros((width*height));
                for i in range((width)):
                    for j in range((height)): #take average zvalue
                       boxpoints[i*height+j] = disparity[int(box[0]+i),int(box[1]+j)]
                    #print(disparity[480])
                boxpoints.sort()
                #dist = disparity[round((box[0]+box[2])/2),round(box[1]+(box[0]+box[2])/2),0]\
                size=boxpoints.size
                result = boxpoints[-round(size/4)]
                result2 = boxpoints[round(size/4)]
                print(result,result2)
                print(confidence)
                print(box)

    cv2.imshow("left",img)
    cv2.waitKey(50)




