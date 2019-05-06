# -*- coding: utf-8 -*-

"""
Version: 3.7.2
Author: Ünver Can Ünlü
"""

import os
import cv2 as cv
import numpy as np

# file paths
PROJECT = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(PROJECT, 'input.mp4')
OUTPUT = os.path.join(PROJECT, 'output.mp4')
YOLO = os.path.join(PROJECT, 'yolov3')
LABELS = os.path.join(YOLO, 'coco.names')
CONFIG = os.path.join(YOLO, 'yolov3.cfg')
WEIGTHS = os.path.join(YOLO, 'yolov3.weights')

# network parameters
MEAN = [0, 0, 0]    # no mean
SCALE = 1 / 255     # interval [0, 1]
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
INPUT_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
DEVICE = cv.dnn.DNN_TARGET_CPU
COMPUTATION = cv.dnn.DNN_BACKEND_OPENCV

def get_output_layers(network):
    """ get output layers of given network """
    output_layers = []
    for unconnected_output_layers in network.getUnconnectedOutLayers():
        layers_names = network.getLayerNames()
        output_layer = layers_names[unconnected_output_layers[0] - 1]
        output_layers.append(output_layer)
    return output_layers

def get_detected_objects(outputs, labels, frame_size, threshold=0.5):
    """ get detected objects from given outputs """
    frame_width, frame_height = frame_size
    detecteds = []
    for detections in outputs:
        for detection in detections:
            label_scores = detection[5:]
            score = float(label_scores[np.argmax(label_scores)])
            if score > threshold:
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(int(detection[0] * frame_width) - width / 2)
                top = int(int(detection[1] * frame_height) - height / 2)
                detected = {
                    'id': np.argmax(label_scores),
                    'label': labels[np.argmax(label_scores)],
                    'score': score,
                    'box': (left, top, width, height)
                }
                detecteds.append(detected)
    return detecteds

def read_labels(label_file):
    """ read labels names from given file """
    labels = []
    with open(label_file, mode='r') as opened_file:
        line = opened_file.readline().rstrip('\n')
        while line:
            labels.append(line)
            line = opened_file.readline().rstrip('\n')
    return labels

def non_maxima_suppression(detecteds, score_threshold=0.5, nms_threshold=0.5):
    """ non-maxima suppression on given detected boxes """
    scores = [detected['score'] for detected in detecteds]
    boxes = [detected['box'] for detected in detecteds]
    indices = cv.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    for index in indices:
        index = index[0]
        detecteds[index]['box'] = boxes[index]

def draw_detecteds(frame, detecteds, color=(255, 255, 255), font=cv.FONT_HERSHEY_PLAIN, size=1):
    """ draw given detecteds on given frame """
    processed = frame.copy()
    for detected in detecteds:
        left, top, width, height = detected['box']
        cv.rectangle(processed, (left, top), (left + width, top + height), color)
        if 'score' in detected:
            text = '{label}: {score:.2f}'.format(label=detected['label'], score=detected['score'])
        else:
            text = detected['label']
        cv.putText(processed, text, (left, top - 10), font, size, color)
    return processed

def main():
    """ YOLOv3 Object Detection on Video """
    print('YOLOv3 Object Detection on Video')
    print('Input video: {input}'.format(input=INPUT))
    print('Output video: {output}'.format(output=OUTPUT))

    # network
    network = cv.dnn.readNetFromDarknet(CONFIG, WEIGTHS)
    network.setPreferableBackend(COMPUTATION)
    network.setPreferableTarget(DEVICE)
    object_labels = read_labels(LABELS)

    # video
    video = cv.VideoCapture(INPUT)
    fps = video.get(cv.CAP_PROP_FPS)
    codec = cv.VideoWriter_fourcc(*'mp4v')
    frame_width = round(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = round(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    writer = cv.VideoWriter(OUTPUT, codec, fps, frame_size)

    # video processing
    print('Video processing is started.')
    frame_count = 0
    progress = 0
    while video.isOpened():
        frame_exist, frame = video.read()
        if frame_exist:
            frame_count = frame_count + 1
            if progress != int(frame_count / total_frames * 100):
                progress = int(frame_count / total_frames * 100)
                print('{progress}/100 is completed.'.format(progress=progress))

            # process frame and detect objects
            blob = cv.dnn.blobFromImage(frame, SCALE, INPUT_SIZE, MEAN, swapRB=True, crop=False)
            network.setInput(blob)
            outputs = network.forward(get_output_layers(network))
            detecteds = get_detected_objects(outputs, object_labels, frame_size)
            if len(detecteds) > 0:
                non_maxima_suppression(detecteds)

            # generate output frame
            if len(detecteds) > 0:
                frame_processed = draw_detecteds(frame, detecteds)
                writer.write(frame_processed.astype(np.uint8))
            else:
                writer.write(frame.astype(np.uint8))
        else:
            video.release()
            writer.release()
            break
    print('Video processing is done.')

if __name__ == "__main__":
    main()
