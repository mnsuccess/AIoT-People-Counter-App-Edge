"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client
 
def draw_boxes(frame,output_result,prob_threshold, width, height):
    """
    Draw bounding boxes onto the frame.
    :param frame: frame from camera/video
    :param result: list contains the data comming from inference
    :return: person count and frame
    """
    counter=0
    # Start coordinate, here (xmin, ymin) 
    # represents the top left corner of rectangle 
    start_point = None
    # Ending coordinate, here (xmax, ymax) 
    # represents the bottom right corner of rectangle 
    end_point = None 
    # Blue color in BGR 
    color = (0, 255, 0)
    # Line thickness of 2 px 
    thickness = 1
    for box in output_result[0][0]: # Output shape is 1x1x100x7
        if box[2] > prob_threshold:
            start_point = (int(box[3] * width), int(box[4] * height))
            end_point = (int(box[5] * width), int(box[6] * height))
            # Using cv2.rectangle() method 
            # Draw a rectangle with Green line borders of thickness of 1 px
            frame = cv2.rectangle(frame, start_point, end_point, color,thickness)
            counter+=1
    return frame, counter
    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Input arguments
    modelArgs = args.model
    deviceArgs = args.device
    cpuExtensionArgs = args.cpu_extension
    propThresholdArgs = args.prob_threshold
    filePathArgs = args.input
    
    # Initialise the class
    infer_network = Network()
    
    #Load the model through `infer_network`
    infer_network.load_model(modelArgs,deviceArgs,cpuExtensionArgs)
    net_input_shape = infer_network.get_input_shape()
    
    # Set Probability threshold for detections
    prob_threshold = propThresholdArgs
    
    # Handle image, video or webcam
    # Create a flag for single images
    # Flag for the input image
    single_image_mode = False
    # Check if the input is a webcam
    if filePathArgs == 'CAM':
        filePathArgs = 0
    elif filePathArgs.endswith('.jpg') or filePathArgs.endswith('.bmp'):
        single_image_mode = True

    # Handle the input stream 
    # Get and open video capture
    capture = cv2.VideoCapture(filePathArgs)
    capture.open(filePathArgs)

    # Grab the shape of the input 
    width = int(capture.get(3))
    height = int(capture.get(4))
    
    # initlise some variable 
    report = 0
    counter = 0
    counter_prev = 0
    duration_prev = 0
    counter_total = 0
    dur = 0
    request_id=0
    
    # Process frames until the video ends, or process is exited
    while capture.isOpened():
        # Read the next frame
        flag, frame = capture.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        #Re-size the frame to inputshape_width x inputshape_height
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        #Start asynchronous inference for specified request
        #Perform inference on the frame
        duration_report = None
        inf_start = time.time()
        infer_network.exec_net(p_frame)
        # Get the output of inference
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start
            # Results of the output layer of the network
            output_results = infer_network.get_output()
            #Extract any desired stats from the results 
            #Update the frame to include detected bounding boxes
            frame_with_box, pointer = draw_boxes(frame, output_results, prob_threshold, width, height)
            #Display inference time
            inf_time_message = "Manasse_Ngudia | Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame_with_box, inf_time_message, (15, 15),
                       cv2.FONT_HERSHEY_COMPLEX, 0.45, (200, 10, 10), 1)
                    
            #Calculate and send relevant information on 
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if pointer != counter:
                counter_prev = counter
                counter = pointer
                if dur >= 3:
                    duration_prev = dur
                    dur = 0
                else:
                    dur = duration_prev + dur
                    duration_prev = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > counter_prev:
                        counter_total += counter - counter_prev
                    elif dur == 3 and counter < counter_prev:
                        duration_report = int((duration_prev / 10.0) * 1000)
                        
            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
            
            #Send frame to the ffmpeg server
            #  Resize the frame
            #frame = cv2.resize(frame, (768, 432))
            sys.stdout.buffer.write(frame_with_box)
            sys.stdout.flush()
            
            if single_image_mode:
                cv2.imwrite('output_image.jpg', frame_with_box)

        # Break if escapturee key pressed
        if key_pressed == 27:
            break
        

    # Release the out writer, captureture, and destroy any OpenCV windows
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
