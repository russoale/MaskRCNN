# -*- coding: utf-8 -*-
"""
Created on 15 Jun 2018, 10:24

@author: einfalmo
"""

import os
import numpy as np
import cv2

from pose_plot import draw_jump_pose
import annotation_io
from video_utils.simple_video_reader import  SimpleVideoReader


if __name__ == "__main__":

    annotation_path = "/home/einfalmo/Documents/Projekte/bisp18/sprung/annotations/complete.csv"
    path_to_videos = "/data_ssd/daten/bisp18_sprung/Sprung/Dreisprung/Videos"

    # Read annotations
    athletes, events, frame_nums, annotations = annotation_io.read_jump_csv_annotations(annotation_path)

    # Look for specific video
    event_index = [i for i in range(len(events)) if events[i] == "drei 180217 Gierisch 5"][0]
    first_event = events[event_index]

    path_to_event_video = os.path.join(path_to_videos, first_event + ".MTS")
    video_reader = SimpleVideoReader(path_to_event_video)

    # Pick first annotated frame in this video
    frame_num = frame_nums[event_index]
    frame = video_reader.at(frame_num)

    # Draw annotated pose onto the image
    draw_jump_pose(frame, annotations[event_index], draw_into_original=True)

    cv2.imshow("", frame[:, :, ::-1])
    cv2.waitKey()
    cv2.destroyAllWindows()

