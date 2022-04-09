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
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import matplotlib.pyplot as plt

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import serial as ser
import serial_bus_servo_controller as sbsc

#Author: Kevin Shao 2022

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


# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)


disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)
CalibratedPair = CalibratedPair(None, calibration, sbm)


my_chain = Chain(name='left_arm', links=[
    OriginLink(),
    URDFLink(
      name="base rotation",
      bounds=(-np.deg2rad(135),np.deg2rad(135)),
      origin_translation=[0, 0, 0], #distance from previous link (gives length of limb between previous and current link)
      origin_orientation=[0, 0, 0], #original orientation rpy
      rotation=[0, 1, 0],
    ),
    URDFLink(
      name="base pitch",
      bounds=(-np.deg2rad(90),np.deg2rad(90)),
      origin_translation=[0, 62, 0], #distanc from baseplate to motor 5
      origin_orientation=[0, 0, 0],
      rotation=[1, 0, 0],
    ),
    URDFLink(
      name="shoulder", #4
      bounds=(-np.deg2rad(130),np.deg2rad(130)),
      origin_translation=[0, 101, 0],
      origin_orientation=[0, 0, 0],
      rotation=[1, 0, 0],
    ),
    URDFLink(
      name="elbow", #3
      bounds=(-np.deg2rad(110),np.deg2rad(110)),
      origin_translation=[0, 96, 0],
      origin_orientation=[0, 0, 0],
      rotation=[1, 0, 0],
    ),
    URDFLink(
      name="wrist", #2
      origin_translation=[0, 58.03, 0], 
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 0], #does not account for rotation of wrist
    ),
    URDFLink(
      name="gripper", #1 #positions range from 300-700 for 1
      origin_translation=[0, 112, 0], 
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 0],
    ),
])

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    Q = np.float32([[3,0,0,width/2.0],[0,5,0,-height/2.0],[0,0,0,-width*0.8],[0,0,7,0]])
    point_cloud = cv2.reprojectImageTo3D(disparity,Q)     
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit();
    return point_cloud

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


load_map_settings ("3dmap_set.txt")

img = cv2.imread('Lenna.png')
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
camera.software_auto_white_balance(enable = True)
#camera.software_auto_exposure(enable = True)

classNames = []
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net= cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(420,420)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

def rad_to_servo_pos(radians):
    servo=500*np.ones(6,int)
    for i in range(len(servo)):
        servo[i] += round(241.9*radians[i+1]) #380 points/90deg = 380/(pi/2)
    servo[3] = 500 - round(241.9*radians[4]) #orientation is flipped
    servo[1] = 500 - round(241.9*radians[2]) #orientation is flipped
    servo[4] = 880
    servo[5] = 700
    return servo

import time
controller = sbsc.SBS_Controller("/dev/ttyUSB0")
frames = 3
x = [-390,-390,-390]   ##tapewise
z = [1,1,1]  ##perpendicular to tape
y = [150,50,50] ##vertical
claw = [300,300,800]
controller.cmd_servo_move([6,5,4,3,2,1], [500,500,500,500,500,700], 2000)
time.sleep(2)
for i in range(frames):
    if y[i] >= 30 and abs(x[i]) < 400 and abs(z[i])<400:
        # plot the line chart     
        result = my_chain.inverse_kinematics([x[i],y[i],z[i]])           
        servo_pos = rad_to_servo_pos(result)
        servo_pos[4] = 500
        servo_pos[5] = claw[i] 
        controller.cmd_servo_move([6,5,4,3,2,1],servo_pos, 1500)
        time.sleep(2)
        print(i,"/",frames, servo_pos)
controller.cmd_servo_move([6,5,4,3,2,1], [500,500,500,500,500,800], 2000)

while True:
    frame = get_frame(camera)
    frame = cv2.resize(frame, (img_width, img_height))
    print(img_width/2,img_height)
    img = frame [0:img_height,0:int(img_width/2)] #Left image Y+H and X+W
    imgL = frame [0:img_height,0:int(img_width/2)] #Left image Y+H and X+W
    cv2.imshow("left",imgL)
    cv2.imwrite("CVimgL.png",imgL)
    imgR = frame [0:img_height,int(img_width/2):img_width] #Left image Y+H and X+W
    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    rectified_pair = calibration.rectify((imgLeft, imgRight)) 
    disparity = stereo_depth_map(rectified_pair)
    # show the frame   

    t2 = datetime.now()
    #print ("DM build time: " + str(t2-t1))
    
    classIds,confs,bbox=net.detect(img,confThreshold=0.6)
    #print(classIds,bbox)
    
    if len(classIds) != 0:
          for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
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
                cv2.rectangle(disparity,boxa,boxb,color=(255,255,0),thickness=6)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0 ,0),2)
                cv2.rectangle(img,startpoint,endpoint,color=(255,255,0),thickness=2)
                cv2.rectangle(disparity,box,color=(255,0,0),thickness=2)
                cv2.putText(disparity,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0 ,0),2)
                cv2.rectangle(disparity,startpoint,endpoint,color=(255,255,0),thickness=2)
                
    #detect length to hand
    length = 4.218726117 # focus length
    if len(classIds) != 0:
        conf = 0
        dist = 70
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classId == 1: #identified as human
                width = box[2]
                height = box[3]
                #print(box)
                #print(img_height,img_width)
                boxpoints = np.zeros((width*height));
                for i in range((width)):
                    for j in range((height)): #take average zvalue
                       boxpoints[i*height+j] = disparity[box[1]+j,box[0]+i,0]
                    #print(disparity[480])
                boxpoints.sort()
                #dist = disparity[round((box[0]+box[2])/2),round(box[1]+(box[0]+box[2])/2),0]
                if boxpoints[50]>-0.6:
                    if conf < confidence:
                        dist = dist = -269*boxpoints[50]-18.5
                        print(dist,confidence)
                        conf = confidence
        if dist < 70:
            #box[0] is x from left, box[1] is y from top
            z = 5+(length+dist)*(2.160128321/(img_width/4))*(box[0]-img_width/4)/length
            y = 10.3-(length+dist)*((img_height/2)/(img_width/4))*(2.160128321/(img_width/4))*(box[1]-img_height/2)/length
            print(box[0],box[1],img_width,img_height)
            print("x,y,z=",dist,y,z)
            distr = 10*dist-440
            yr = 10*y+100#15cm for safety
            zr = 10*z+200 #-10cm for safety
            if yr >= 5 and abs(distr) < 300 and abs(zr)<300:   
                result = my_chain.inverse_kinematics([distr,yr,zr])           
                servo_pos = rad_to_servo_pos(result)
                servo_pos[4] = 500
                servo_pos[5] = 800
                controller.cmd_servo_move([6,5,4,3,2,1],servo_pos, 1000)
                time.sleep(1)
                

                

    cv2.imshow("Image", disparity)
    cv2.imshow("Output",img)
    cv2.imwrite("CVimg.png",img)
    cv2.imshow("right",imgR)
    cv2.imwrite("CVimgR.png",imgR)
    cv2.waitKey(50)



  