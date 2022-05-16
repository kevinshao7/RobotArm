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
camera.set_mode(12)
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
      bounds=(-np.deg2rad(80),np.deg2rad(80)),
      origin_translation=[0, 62, 0], #distanc from baseplate to motor 5
      origin_orientation=[0, 0, 0],
      rotation=[1, 0, 0],
    ),
    URDFLink(
      name="shoulder", #4
      bounds=(-np.deg2rad(110),np.deg2rad(110)),
      origin_translation=[0, 101, 0],
      origin_orientation=[0, 0, 0],
      rotation=[1, 0, 0],
    ),
    URDFLink(
      name="elbow", #3
      bounds=(-np.deg2rad(80),np.deg2rad(80)),
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
      origin_translation=[0, 100, 0], 
      origin_orientation=[0, 0, 0],
      rotation=[0, 0, 0],
    ),
])

img = cv2.imread('Lenna.png')
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
camera.software_auto_white_balance(enable = True)
camera.software_auto_exposure(enable = True)



classNames = []
classFile='obj.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(img_width,img_height)
configPath = 'yolov4-custom.cfg'
weightsPath = 'yolov4-custom_7000.weights'

net= cv2.dnn.readNetFromDarknet(configPath,weightsPath)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

def rad_to_servo_pos(radians):
    servo=500*np.ones(6,int)
    for i in range(len(servo)):
        servo[i] += round(241.9*radians[i+1]) #380 points/90deg = 380/(pi/2)
    servo[3] = 500 - round(241.9*radians[4]) #orientation is flipped
    servo[1] = 500 - round(241.9*radians[2]) #orientation is flipped
    servo[4] = 500
    servo[5] = 300
    return servo

import time

controller = sbsc.SBS_Controller("/dev/ttyUSB0")
controller.cmd_servo_move([6,5,4,3,2,1], [500,500,500,500,500,700], 3000)
time.sleep(4)


def checkbounds(goal,duration):
  if((goal[0] > 100 & goal[0]<900)&
  (goal[1] > 200 & goal[1]<800)&
  (goal[2] > 200 & goal[2]<800)&
  (goal[3] > 200 & goal[3]<800)&
  (goal[4] > 100 & goal[4]<900)&
  (goal[5] > 100 & goal[5]<900)
  ):
    controller.cmd_servo_move([6,5,4,3,2,1],goal, duration)

def closeenough(error):
  close = True
  for i in range(4):
    if (error[i]>2 or error[i]<-2):
      close=False
  return close


def move(destination,duration):
  print('p1')
  dest = np.array(destination)
  checkbounds(destination,duration)  
  time.sleep(round(0.2+duration/1000))
  real = np.array(controller.cmd_mult_servo_pos_read([6,5,4,3,2,1]))
  real = np.array(controller.cmd_mult_servo_pos_read([6,5,4,3,2,1]))
  real = np.array(controller.cmd_mult_servo_pos_read([6,5,4,3,2,1]))
  discrepancy = destination-real
  print(discrepancy)
  print(np.any(discrepancy != np.array([0,0,0,0,0,0])))

  while (not closeenough(discrepancy)):
    print(destination,discrepancy)
    dest += discrepancy
    checkbounds(dest,300)
    time.sleep(0.5)
    real = controller.cmd_mult_servo_pos_read([6,5,4,3,2,1])
    discrepancy = destination-real
  print("done")

bins = [[260,80,120],[160,80,20],[260,80,20]]
while True:
    frame = get_frame(camera)
    frame = cv2.resize(frame, (img_width, img_height))
    img = frame [0:img_height,0:int(img_width/2)] #Left image Y+H and X+W
    t1 = datetime.now()
    # show the frame   

    t2 = datetime.now()
    #print ("DM build time: " + str(t2-t1))
    
    classIds,confs,bbox=model.detect(img,confThreshold=0.9)
    #print(classIds,bbox)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(img,box,color=(255,0,0),thickness=2)
                cv2.putText(img,classNames[classId].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0 ,0),2)
                print(box)
    cv2.imshow("Output",img)
    cv2.waitKey(50)
    #detect length to hand
    ratio = 0.1203125 # convert pixels to cm
    print("here")
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            if box[1]+box[3] < img_height/2: #above halfway
                print("classId",classId)
                cv2.rectangle(img,box,color=(255,0,0),thickness=2)
                cv2.putText(img,classNames[classId].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,
                            (255,0 ,0),2)
                width = box[2]
                height = box[3]
                #print(box)
                #print(img_height,img_width)
                imgx = (box[0]+box[2]/2)-img_width/4
                imgy = (box[1] + box[3])-img_height/2 #target bottom center
                realX = -imgx*ratio
                realY = -imgy*ratio
                armX = round(realX*10)+45
                armZ = round(realY*10)+110-round(1.5*box[3]*(imgy-img_height/2)/img_height)
                #box[0] is x from left, box[1] is y from top
                print("box=",box)
                print("x,y,z=",realX,15,realY)  
                coord = [armX,5,armZ]
                precoord = [armX,104,armZ]
                preresult = my_chain.inverse_kinematics(precoord)    
                result = my_chain.inverse_kinematics(coord)
                sortbin = my_chain.inverse_kinematics(bins[classId])
                servo_pos = rad_to_servo_pos(result)
                bin_pos = rad_to_servo_pos(sortbin)
                bin_pos[5]=700
                preservo_pos = rad_to_servo_pos(preresult)
                move(preservo_pos, 1000)    
                move(servo_pos, 500)
                servo_pos[5] = 700
                preservo_pos[5]=700
                move(servo_pos, 500)
                move(preservo_pos, 500)
                move(bin_pos,1000)
                bin_pos[5]=300
                move(bin_pos,500)
                time.sleep(1)
                break
                    

                


    cv2.imshow("Image",img)



  

