# RobotArm
Kevin Shao
TVSEF 2022
April 9, 2022
Code For A Computer Vision-Controlled Smart Robotic Arm
# Requirements
 - Arducam Stereo Camera
 - Raspberry Pi 4
 - Hiwonder Xarm 1s robotic arm
 - USB to TTL serial adapter
# Installation Instructions
1. Follow the hardware and software setup instructions for the stereo camera [here](https://www.arducam.com/docs/cameras-for-raspberry-pi/synchronized-stereo-camera-hat/opencv-and-depth-map-on-arducam-stereo-camera-hat-tutorial/). The website will walk you through calibration steps. Set up the environment on the Raspberry Pi. Follow the instructions with respect to python package prerequisites. At the 5_dm_tune.py stage, calibrate the stereo module to be as stable and noiseless as possible.
***Note that stereo vision (depth perception) is not currently enabled in this iteration of the project***

3. Enter the folder /home/pi/MIPI_Camera/RPI/stereo_depth_demo.
4. Download the contents of this repo into the specified folder above, specifically:
 - sortlego.py
 - serial_bus_servo_control.py
 - yolov4-custom.cfg
 - obj.names
 - camera_params.txt
 - Additionally, download [the weights](https://drive.google.com/file/d/1-461Gj4L3T8d_dMuW-LbIDLgsFdhNf0F/view?usp=sharing)
5. Set up the camera and roboti arm in the appropriate positions (details will follow soon).
6. Plug in the USB to TTL serial adapter into the Raspberry Pi. Connect the ground of the adapter to the corresponding pin on the robotic arm. Connect the transmit pin of the adapter to the receiving pin on the robotic arm. Connect the receive pin of the adapter to the transmit pin on the robotic arm.
7. Run sortlego.py.
