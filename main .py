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
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def SingleSD(frame, result, initial_w, initial_h):
    
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 5), 1)
            current_count = current_count + 1
    return frame, current_count


def infer_on_stream(args, client):

    infer_network = Network()
    
         
    single_input_mode = False
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0    
   # Set Probability threshold for detections
    global prob_threshold

    prob_threshold = args.prob_threshold
    model = args.model
    
    device = args.device
    cpu_extension = args.cpu_extension             
    inlm = infer_network.load_model(model,cpu_extension=cpu_extension, device="CPU")
    
    network_shape = infer_network.get_input_shape()
    
    # Check for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Check for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Check for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
    
    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(input_stream)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    
    initial_w = int(cap.get(3))
    initial_h = int(cap.get(4))
   
        #Read from the video capture #
   
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)    
        
        ### TODO: Pre-process the image as needed ###
        image_p = cv2.resize(frame,(network_shape[3], network_shape[2]))
        image_p = image_p.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)

        inf_start = time.time()
        infer_network.exec_net(0,image_p)        
    
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start

            net_output = infer_network.get_output()

            ### Update the frame to include detected bounding boxes
        frame, current_count = SingleSD(frame, net_output,initial_w, initial_h)
        inf_time_message = "Inf time: {:.3f}ms".format(det_time * 1000)
        cv2.putText(frame,inf_time_message,(15, 15),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
    
    
                              
        if current_count > last_count:  # New entry
            start_time = time.time()
            diff = current_count - last_count
            total_count = total_count + diff
            client.publish("person", json.dumps({"total": total_count}))

        if current_count < last_count:  # Average Time
            duration = int(time.time() - start_time)
            client.publish("person/duration", json.dumps({"duration": duration}))

        if current_count > 1:
            client.publish("person", json.dumps({"count": 1}))
        else:
            client.publish("person", json.dumps({"count": current_count}))
                  # People Count

            last_count = current_count
            if key_pressed == 27:
                break
            
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        # Write an output image if `single_image_mode` ###
       
    if single_image_mode:
        cv2.imwrite('output_image.jpg', frame)
        
        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

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

    