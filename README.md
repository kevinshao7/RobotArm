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
1. Follow the hardware and software setup instructions for the stereo camera [here](https://www.arducam.com/docs/cameras-for-raspberry-pi/synchronized-stereo-camera-hat/opencv-and-depth-map-on-arducam-stereo-camera-hat-tutorial/). The website will walk you through calibration steps. Set up the environment on the Raspberry Pi. Follow the instructions with respect to python package prerequisites.
2. Enter the folder /home/pi/MIPI_Camera/RPI/stereo_depth_demo.
3. Download the contents of this repo into the specified folder above, specifically:
 - dropball.py
 - serial_bus_servo_control.py
 - ssd_mobilenet_v3
 - frozen_inference_graph.pb
 - coco.names
4. Plug in the USB to TTL serial adapter into the Raspberry Pi. Connect the ground of the adapter to the corresponding pin on the robotic arm. Connect the transmit pin of the adapter to the receiving pin on the robotic arm.
5. Run dropball.py.
