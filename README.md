# RobotArm
Kevin Shao
TVSEF 2022
April 9, 2022
Code For A Computer Vision and Stereo Vision-Controlled Smart Robotic Arm
# Requirements
 - Arducam Stereo Camera
 - Raspberry Pi 4
 - Hiwonder Xarm 1s robotic arm
 - USB to TTL serial adapter
# Installation Instructions
1. Follow the hardware and software setup instructions for the stereo camera [here](https://www.arducam.com/docs/cameras-for-raspberry-pi/synchronized-stereo-camera-hat/opencv-and-depth-map-on-arducam-stereo-camera-hat-tutorial/). The website will walk you through calibration steps. Set up the environment on the Raspberry Pi. Follow the instructions with respect to python package prerequisites. Skip the 5_dm_tune.py stage, and use the camera_params.txt file provided in this repo. 
2. Enter the folder /home/pi/MIPI_Camera/RPI/stereo_depth_demo.
3. Download the contents of this repo into the specified folder above, specifically:
 - dropball.py
 - serial_bus_servo_control.py
 - ssd_mobilenet_v3
 - frozen_inference_graph.pb
 - coco.names
 - camera_params.txt
4. All modules have been calibrated for the setup described below. Instructions on how to recalibrate the system for a generalized setup will follow soon.
   In (x,z) coordinates (representing the surface of a table), measured in centimetres:
       - Define the positive and negative z direction as east and west respectively
       - Define the positive and negative x direction as north and south respectively
       - Set the southwest corner of the robotic arm baseplate at (32cm, 12cm). Ensure that the square baseplate is parallel with the coordinate axes
       - Centralize the stereo camera at (-14cm,0cm). Fix the lenses at 10cm height looking horizontally east
6. Plug in the USB to TTL serial adapter into the Raspberry Pi. Connect the ground of the adapter to the corresponding pin on the robotic arm. Connect the transmit pin of the adapter to the receiving pin on the robotic arm.
7. Run dropball.py.
