#!/usr/bin/env python3
"""This module contains the Yolo class
that uses the Yolo v3 algorithm to perform object detection
includes processing output method
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.activations import sigmoid  # type: ignore


class Yolo:
    """This class uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Yolo class constructor
        Args:
            model_path: path to a Darknet Keras model
            classes_path: path to list of class names
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs: list of numpy.ndarrays with predictions
            image_size: numpy.ndarray with original image size
        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = output[..., :4]
            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            c_x = np.arange(grid_width).reshape(1, grid_width)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            c_y = np.arange(grid_width).reshape(1, grid_width)
            c_y = np.repeat(c_y, grid_height, axis=0).T
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            b_x = (sigmoid(t_x) + c_x) / grid_width
            b_y = (sigmoid(t_y) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            x1, y1 = (b_x - b_w / 2), (b_y - b_h / 2)
            x2, y2 = (b_x + b_w / 2), (b_y + b_h / 2)

            x1 = x1 * image_size[1]
            y1 = y1 * image_size[0]
            x2 = x2 * image_size[1]
            y2 = y2 * image_size[0]

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

            box_confidence = output[..., 4:5]
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_confidences.append(box_confidence)

            box_class_prob = output[..., 5:]
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """function that filters boxes based on their objectness score"""
        box_scores = []
        box_classes = []
        filtered_boxes = []

        for i, (box_confidence,
                box_class_prob, box) in enumerate(zip(box_confidences,
                                                      box_class_probs, boxes)):
            box_scores_per_ouput = box_confidence * box_class_prob
            max_box_scores = np.max(box_scores_per_ouput, axis=3)
            max_box_scores = max_box_scores.reshape(-1)
            max_box_classes = np.argmax(box_scores_per_ouput, axis=3)
            max_box_classes = max_box_classes.reshape(-1)
            box = box.reshape(-1, 4)

            index_list = np.where(max_box_scores < self.class_t)
            max_box_scores_filtered = np.delete(max_box_scores, index_list)
            max_box_classes_filtered = np.delete(max_box_classes, index_list)
            filtered_box = np.delete(box, index_list, axis=0)

            box_scores.append(max_box_scores_filtered)
            box_classes.append(max_box_classes_filtered)
            filtered_boxes.append(filtered_box)

        box_scores = np.concatenate(box_scores)
        box_classes = np.concatenate(box_classes)
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)

        return (filtered_boxes, box_classes, box_scores)
