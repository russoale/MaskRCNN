#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:12:33 2016

@author: einfalmo
"""

import os
import numpy as np
import csv
from samples.jump.bisp_joint_order import JumpJointOrder, SwimJointOrder

# CSV header for annotation I/O
_swimming_annotation_header = ["athlete", "event", "frame_num", "cam_num"] + [joint_name + suffix for joint_name in SwimJointOrder.names() for suffix in ["_x", "_y", "_vis"]]
_swimming_channel_annotation_header = ["image"] + [joint_name + suffix for joint_name in SwimJointOrder.names() for suffix in ["_x", "_y", "_vis"]]
_jump_annotation_header = ["athlete", "event", "frame_num"] + [joint_name + suffix for joint_name in JumpJointOrder.names() for suffix in ["_x", "_y", "_vis"]]


# CSV header for bbox annotation I/O
_swimming_bbox_annotation_header = ["athlete", "event", "frame_num", "cam_num", "x", "y", "width", "height"]



class SkipCSVCommentsIterator:
    """
    Simple file-iterator wrapper to skip empty and '#'-prefixed lines.
    Taken from https://bytes.com/topic/python/answers/513222-csv-comments
    (User: skip)
    """
    
    def __init__(self, fp):
        self.fp = fp

    def __iter__(self):
        return self
    
    def __next__(self):
        line = next(self.fp)
        if not line.strip() or line[0] == "#":
            return next(self)
        return line


def read_swimming_csv_annotations(csv_path, num_joints=SwimJointOrder.num_joints):
    """
    Reads annotations from the given csv file. Format is expected to be the "(4 + 3*14) entries per row" one.
    The first line has to be a comment with the number of annotations, eg. "#123"
    
    Parameters:
        - csv_path: Path to the csv file
        
    returns_
        - List of athlete names
        - List of event names
        - List of frame numbers
        - List of camera numbers
        - Numpy.ndarray (num_images x 14 x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    athletes = list()
    events = list()
    frame_nums = list()
    cam_nums = list()
    joints = np.ndarray(shape=(num_annotations, num_joints, 3), dtype=np.int)
        
    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert(len(row) == 4 + 3*num_joints)
            athletes.append(row[0])
            events.append(row[1])
            frame_nums.append(int(row[2]))
            cam_nums.append(int(row[3]))
            joint_count = 0
            for i in range(6, len(row), 3):
                joints[row_count, joint_count] = [float(row[i-2]), float(row[i-1]), float(row[i])]
                joint_count += 1
            row_count += 1
            
    return athletes, events, frame_nums, cam_nums, joints


def read_swimming_channel_csv_annotations(csv_path, num_joints=SwimJointOrder.num_joints):
    """
    Reads annotations from the given csv file. Format is expected to be the "(1 + 3*14) entries per row" one.
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file

    returns_
        - List of image names (relative path inside of the dataset)
        - Numpy.ndarray (num_images x 14 x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    image_names = list()
    joints = np.ndarray(shape=(num_annotations, num_joints, 3), dtype=np.int)

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert (len(row) == 1 + 3 * num_joints)
            image_names.append(row[0])
            joint_count = 0
            for i in range(3, len(row), 3):
                joints[row_count, joint_count] = [float(row[i - 2]), float(row[i - 1]), float(row[i])]
                joint_count += 1
            row_count += 1

    return image_names, joints


def read_jump_csv_annotations(csv_path, num_joints=JumpJointOrder.num_joints):
    """
    Reads annotations from the given csv file. Format is expected to be the "(3 + 3*20) entries per row" one.
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file

    returns_
        - List of athlete names
        - List of event names
        - List of frame numbers
        - List of camera numbers
        - Numpy.ndarray (num_images x 20 x 3) with the joint locations and the "visibility" flag.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    athletes = list()
    events = list()
    frame_nums = list()
    joints = np.ndarray(shape=(num_annotations, num_joints, 3), dtype=np.int)

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert (len(row) == 3 + 3 * num_joints)
            athletes.append(row[0])
            events.append(row[1])
            frame_nums.append(int(row[2]))
            joint_count = 0
            for i in range(5, len(row), 3):
                joints[row_count, joint_count] = [float(row[i - 2]), float(row[i - 1]), float(row[i])]
                joint_count += 1
            row_count += 1

    return athletes, events, frame_nums, joints


def read_swimming_box_csv_annotations(csv_path, num_joints=SwimJointOrder.num_joints):
    """
    Reads annotations from the given csv file. Format is expected to be the "(4 + 4) entries per row" one.
    The first line has to be a comment with the number of annotations, eg. "#123"

    Parameters:
        - csv_path: Path to the csv file

    returns_
        - List of athlete names
        - List of event names
        - List of frame numbers
        - List of camera numbers
        - Numpy.ndarray (num_images x 4) with the bounding box annotations (in x,y,w,h format).
    """

    with open(csv_path, "r") as f:
        first_line = f.readline()
        num_annotations = int(first_line.strip("#").strip("\n").strip("\r").strip(";"))

    athletes = list()
    events = list()
    frame_nums = list()
    cam_nums = list()
    boxes = np.ndarray(shape=(num_annotations, 4), dtype=np.int)

    with open(csv_path, "r") as f:
        reader = csv.reader(SkipCSVCommentsIterator(f), delimiter=';')
        row_count = 0
        skip_header = True
        for row in reader:
            if skip_header:
                skip_header = False
                continue
            assert (len(row) == 4 + 4)
            athletes.append(row[0])
            events.append(row[1])
            frame_nums.append(int(row[2]))
            cam_nums.append(int(row[3]))
            for i in range(0, 4):
                boxes[row_count, i] = int(row[4 + i])
            row_count += 1

    return athletes, events, frame_nums, cam_nums, boxes



            
def write_swimming_csv_annotations(csv_path, athlete_list, event_list, frame_num_list, cam_num_list, annotations):
    """
    Writes the joint annotations for the given video frames into a csv file.
    Format is 4 + 3*14 entries per row.
    First line contains comment with the number of annotations.
    Second line is the csv header.
    
    Parameters:
        - csv_path: File path to write to.
        - athlete_list: List of athletes the video frames belong to
        - event_list: List of event the video frames belong to
        - frame_num_list: List of frame numbers
        - cam_num_list: List of camera numbers the video frames belong to
        - annotations: numpy.ndarray
            Annotations of size (num_images x num_joints * 3)
    """
    with open(csv_path, "w") as f:
        f.write("#%d\n" % annotations.shape[0])
        writer = csv.writer(f, delimiter=';')
        writer.writerow(_swimming_annotation_header)
        for i, annotation in enumerate(annotations):
            row = [athlete_list[i], event_list[i], frame_num_list[i], cam_num_list[i]] + annotation.flatten().tolist()
            writer.writerow(row)


def write_swimming_channel_csv_annotations(csv_path, image_names, annotations):
    """
    Writes the joint annotations for the given image names into a csv file.
    Format is 1 + 3*14 entries per row.
    First line contains comment with the number of annotations.
    Second line is the csv header.

    Parameters:
        - csv_path: File path to write to.
        - image_names: List of image_names the annotations belong to
        - annotations: numpy.ndarray
            Annotations of size (num_images x num_joints * 3)
    """
    with open(csv_path, "w") as f:
        f.write("#%d\n" % annotations.shape[0])
        writer = csv.writer(f, delimiter=';')
        writer.writerow(_swimming_channel_annotation_header)
        for i, annotation in enumerate(annotations):
            row = [image_names[i]] + annotation.flatten().tolist()
            writer.writerow(row)


def write_jump_csv_annotations(csv_path, athlete_list, event_list, frame_num_list, annotations):
    """
    Writes the joint annotations for the given video frames into a csv file.
    Format is 3 + 3*20 entries per row.
    First line contains comment with the number of annotations.
    Second line is the csv header.

    Parameters:
        - csv_path: File path to write to.
        - athlete_list: List of athletes the video frames belong to
        - event_list: List of events the video frames belong to
        - frame_num_list: List of frame numbers
        - annotations: numpy.ndarray
            Annotations of size (num_images x num_joints * 3)
    """
    with open(csv_path, "w") as f:
        f.write("#%d\n" % annotations.shape[0])
        writer = csv.writer(f, delimiter=';')
        writer.writerow(_jump_annotation_header)
        for i, annotation in enumerate(annotations):
            row = [athlete_list[i], event_list[i], frame_num_list[i]] + annotation.flatten().tolist()
            writer.writerow(row)


def write_swimming_box_csv_annotations(csv_path, athlete_list, event_list, frame_num_list, cam_num_list, boxes):
    """
    Writes the bounding box annotations for the given video frames into a csv file.
    Format is 4 + 4 entries per row.
    First line contains comment with the number of annotations.
    Second line is the csv header.

    Parameters:
        - csv_path: File path to write to.
        - athlete_list: List of athletes the video frames belong to
        - event_list: List of event the video frames belong to
        - frame_num_list: List of frame numbers
        - cam_num_list: List of camera numbers the video frames belong to
        - annotations: numpy.ndarray
            Box annotations of size (num_images x 4) i.e. (x,y,w,h)
    """
    with open(csv_path, "w") as f:
        f.write("#%d\n" % boxes.shape[0])
        writer = csv.writer(f, delimiter=';')
        writer.writerow(_swimming_bbox_annotation_header)
        for i, annotation in enumerate(boxes):
            row = [athlete_list[i], event_list[i], frame_num_list[i], cam_num_list[i]] + annotation.flatten().tolist()
            writer.writerow(row)
            

if __name__ == "__main__":
    path = os.path.abspath("/home/einfalmo/data_ssd_link/daten/bisp18_schwimmer/Annotationen/start.csv")
    athletes, events, frame_nums, cam_nums, joint_annotations = read_swimming_csv_annotations(path)
    # athletes, events, frame_nums and com_nums contain the 
    # athlete, event, frame number and camera number of each annotated frame
    # joint_annotations contains the actual annotations in a single numpy array of shape (#annotated-frames x #joints x 3).
    # For each joint, the annotation consist of the x- and y-coordinate and the "visible"-flag
    print(joint_annotations.shape)
    
    # To get only the annotations for a specific event and a specific camera:
    selected_indices = [i for i in range(joint_annotations.shape[0]) if events[i] == "2017-04-25_17-44Brandauer" and cam_nums[i] == 2]
    selected_annotations = joint_annotations[selected_indices]
    print(len(selected_indices))
    print(selected_annotations.shape)
    
            