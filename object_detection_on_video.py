# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import numpy as np

# file path
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# default parameters
PARAMETER = {
    'mean': [0, 0, 0],  # no mean
    'scale': 1 / 255,  # interval [0, 1]
    'input': {
        'width': 416,
        'height': 416,
    },
    'device': cv2.dnn.DNN_TARGET_CPU,
    'computation': cv2.dnn.DNN_BACKEND_OPENCV,
    'input_video': os.path.join(PROJECT_FOLDER, 'input.mp4'),
    'output_video': os.path.join(PROJECT_FOLDER, 'output.mp4'),
    'yolo': {
        'labels': os.path.join(os.path.join(PROJECT_FOLDER, 'yolov3'), 'coco.names'),
        'config': os.path.join(os.path.join(PROJECT_FOLDER, 'yolov3'), 'yolov3.cfg'),
        'weights': os.path.join(os.path.join(PROJECT_FOLDER, 'yolov3'), 'yolov3.weights'),
    },
    'threshold': {
        'score': 0.5,
        'nms': 0.5
    },
    'color': {
        'text': (255, 255, 255),
        'rectangle': (255, 255, 255)
    },
    'text_font': cv2.FONT_HERSHEY_PLAIN,
    'text_size': 1
}


# main function
def main():
    # settings
    argument_parser = argparse.ArgumentParser()

    # create arguments
    argument_parser.add_argument('-i', '--input',
                                 type=str,
                                 default=PARAMETER['input_video'],
                                 help='path to input video file')

    argument_parser.add_argument('-o', '--output',
                                 type=str,
                                 default=PARAMETER['output_video'],
                                 help='path to output video file')

    # parse arguments
    arguments = argument_parser.parse_args()

    # detect objects on video
    detect_objects_on_video(input_video=arguments.input, output_video=arguments.output)


# detect objects on video using YOLOv3
def detect_objects_on_video(input_video=PARAMETER['input_video'], output_video=PARAMETER['input_video']):
    # show info
    print('YOLOv3 Object Detection on Video:')
    print('input video: "{input_video}"'.format(input_video=input_video))
    print('output video: "{output_video}"'.format(output_video=output_video))

    # check input video file path exists
    if not os.path.exists(input_video):
        print('"{input_video}" does not exist.'.format(input_video=input_video))
        return

    # network
    network = cv2.dnn.readNetFromDarknet(PARAMETER['yolo']['config'], PARAMETER['yolo']['weights'])
    network.setPreferableBackend(PARAMETER['computation'])
    network.setPreferableTarget(PARAMETER['device'])
    labels = read_labels(label_file_path=PARAMETER['yolo']['labels'])

    # capture video
    video = cv2.VideoCapture(input_video)
    fps = video.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(output_video, codec, fps, frame_size)

    # video processing
    print('Object detection on video is started.')
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
            input_size = (PARAMETER['input']['width'], PARAMETER['input']['height'])
            blob = cv2.dnn.blobFromImage(frame, PARAMETER['scale'], input_size, PARAMETER['mean'],
                                         swapRB=True, crop=False)
            network.setInput(blob)
            outputs = network.forward(get_output_layers(network))
            detected_objects = get_detected_objects(outputs=outputs, labels=labels, frame_size=frame_size,
                                                    score_threshold=PARAMETER['threshold']['score'])
            if len(detected_objects) > 0:
                non_maxima_suppression(detected_objects=detected_objects,
                                       score_threshold=PARAMETER['threshold']['score'],
                                       nms_threshold=PARAMETER['threshold']['nms'])

            # generate output frame
            if len(detected_objects) > 0:
                frame_processed = draw_detected_objects(frame=frame, detected_objects=detected_objects,
                                                        rectangle_color=PARAMETER['color']['rectangle'],
                                                        text_color=PARAMETER['color']['text'],
                                                        text_font=PARAMETER['text_font'],
                                                        text_size=PARAMETER['text_size'])
                writer.write(frame_processed.astype(np.uint8))
            else:
                writer.write(frame.astype(np.uint8))
        else:
            video.release()
            writer.release()
            break
    print('Object detection on video is done.')


# get output layers of given network
def get_output_layers(network):
    output_layers = []
    for unconnected_output_layers in network.getUnconnectedOutLayers():
        layers_names = network.getLayerNames()
        output_layer = layers_names[unconnected_output_layers - 1]
        output_layers.append(output_layer)
    return output_layers


# get detected objects from given outputs
def get_detected_objects(labels, frame_size, outputs=[], score_threshold=PARAMETER['threshold']['score']):
    frame_width, frame_height = frame_size
    detected_objects = []
    for detections in outputs:
        for detection in detections:
            label_scores = detection[5:]
            score = float(label_scores[np.argmax(label_scores)])
            if score > score_threshold:
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
                detected_objects.append(detected)
    return detected_objects


# read labels names from given file
def read_labels(label_file_path=PARAMETER['yolo']['labels']):
    labels = []
    with open(label_file_path, mode='r') as opened_file:
        line = opened_file.readline().rstrip('\n')
        while line:
            labels.append(line)
            line = opened_file.readline().rstrip('\n')
    return labels


# non-maxima suppression on given detected boxes
def non_maxima_suppression(detected_objects, score_threshold=PARAMETER['threshold']['score'],
                           nms_threshold=PARAMETER['threshold']['nms']):
    scores = [detected['score'] for detected in detected_objects]
    boxes = [detected['box'] for detected in detected_objects]
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    for index in indices:
        detected_objects[index]['box'] = boxes[index]


# draw given detected objects on given frame
def draw_detected_objects(frame, detected_objects=[], rectangle_color=PARAMETER['color']['rectangle'],
                          text_color=PARAMETER['color']['text'], text_font=PARAMETER['text_font'],
                          text_size=PARAMETER['text_size']):
    processed = frame.copy()
    for detected in detected_objects:
        left, top, width, height = detected['box']
        cv2.rectangle(processed, (left, top), (left + width, top + height), rectangle_color)
        if 'score' in detected:
            text = '{label}: {score:.2f}'.format(label=detected['label'], score=detected['score'])
        else:
            text = detected['label']
        cv2.putText(processed, text, (left, top - 10), text_font, text_size, text_color)
    return processed


# main
if __name__ == '__main__':
    main()
