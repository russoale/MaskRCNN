############################################################
#  Test
############################################################
import time
from unittest import TestCase

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random

import mrcnn.model as modellib
from mrcnn import data_generator, visualize
from mrcnn.load_weights import load_weights
from samples.coco import coco

DATA_DIR = os.path.abspath("/data/hdd")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(DATA_DIR, "russales", "test_logs")

# mask only
COCO_MODEL_PATH_MASK = os.path.join(DATA_DIR, "russales", "logs", "coco20190402T2205", "mask_rcnn_coco_0160.h5")

# keypoint only
COCO_MODEL_PATH_KEYPOINT = os.path.join(DATA_DIR, "russales", "logs", "coco20190404T2343", "mask_rcnn_coco_0160.h5")

# keypoint and mask
COCO_MODEL_PATH = os.path.join(DATA_DIR, "russales", "logs", "coco20190324T2219", "mask_rcnn_coco_0160.h5")

# path to MSCOCO dataset
COCO_DIR = os.path.join(DATA_DIR, "Datasets", "MSCOCO_2017")

# Example Video Path
VIDEO_PATH = os.path.join(DATA_DIR, "russales", "JumpDataset", "Videos", "Dreisprung", "drei 180617 Garritsen 1.mp4")
VIDEO_PATH_OUT = os.path.join(DATA_DIR, "russales", "drei_180617_Garritsen_1_out_160_coco_kp_seg_2.avi")


class CocoTests(TestCase):

    def get_model_path(self, training_heads):
        if training_heads is None:
            return COCO_MODEL_PATH
        elif training_heads == "keypoint":
            return COCO_MODEL_PATH_KEYPOINT
        elif training_heads == "mask":
            return COCO_MODEL_PATH_MASK
        else:
            raise Exception("training_heads not specified correctly")

    def load_test_config(self, training_heads):
        class TestConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            TRAINING_HEADS = training_heads

        config = TestConfig()
        config.display()
        return config

    def load_dataset(self, set="val", task_type="person_keypoints"):
        dataset = coco.CocoDataset(task_type=task_type)
        dataset.load_coco(COCO_DIR, set, year="2017", class_ids=[1], auto_download=False)
        dataset.prepare()

        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
        return dataset

    def test_predict(self):
        # load image and ground truth data
        training_heads = "mask"
        config = self.load_test_config(training_heads)
        dataset = self.load_dataset("val")
        model_path = self.get_model_path(training_heads)

        # load model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

        # Load weights
        print("Loading weights ", model_path)
        load_weights(model, model_path, by_name=True, include_optimizer=False)

        dg = data_generator.DataGenerator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)

        image_ids = random.sample(list(dataset.image_ids), 5)

        rows = math.ceil(len(image_ids) / 5)
        height = (len(image_ids) / 5) * 16
        f = plt.figure(figsize=(80, height))
        for idx, image_id in enumerate(image_ids):
            info = dataset.image_info[image_id]
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))

            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_keypoints, gt_mask_train = \
                dg.load_image_gt(image_id, augmentation=None, use_mini_mask=False)

            if training_heads is None:
                results = model.detect([image], verbose=1)
            elif training_heads == "keypoint":
                results = model.detect_keypoint([image], verbose=1)
            else:
                results = model.detect_mask([image], verbose=1)


            # Display results
            r = results[0]

            bboxes_res = r['bboxes']
            class_ids_res = r['class_ids']
            scores_res = r['scores']
            if training_heads == "keypoint" or training_heads is None:
                keypoints_res = r['keypoints']
            if training_heads == "mask" or training_heads is None:
                mask_res = r['masks']
                print("Mask detected: ", mask_res.max())

            ax = f.add_subplot(rows, 5, idx + 1)

            visualize.display_instances(image, bboxes_res, mask_res, class_ids_res, dataset.class_names, ax=ax)
            # visualize.display_keypoints(image, bboxes_res, keypoints_res, class_ids_res, dataset.class_names,
            #                             skeleton=dataset.skeleton, scores=scores_res, ax=ax, dataset=dataset)

        # plt.show()
        # plt.savefig("coco20190324T2219/160/val/{}-{}.png".format(info["source"], info["id"]))
        plt.savefig("coco20190402T2205/val_multiple_mask_only2.png", bbox_inches='tight')
        # plt.savefig("coco20190324T2219/vid_sequence_Nogueira_train_25.png", bbox_inches='tight')

    def test_video(self):
        # load image and ground truth data
        config = self.load_test_config(None)
        dataset = self.load_dataset("val")
        model_path = self.get_model_path(None)

        # load model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

        # Load weights
        print("Loading weights ", model_path)
        load_weights(model, model_path, by_name=True, include_optimizer=False)

        class_names = ['BG', 'person']

        def _cv2_display_keypoint(image, boxes, keypoints, masks, class_ids, scores, class_names,
                                  skeleton=dataset.skeleton):
            # Number of persons
            N = boxes.shape[0]
            if not N:
                print("\n*** No persons to display *** \n")
            else:
                assert N == keypoints.shape[0] and N == class_ids.shape[0] and N == scores.shape[0], \
                    "shape must match: boxes,keypoints,class_ids, scores"
            colors = visualize.random_colors(N)
            for i in range(N):
                color = colors[i]
                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
                for Joint in keypoints[i]:
                    if Joint[2] != 0:
                        cv2.circle(image, (Joint[0], Joint[1]), 2, color, -1)

                # draw skeleton connection
                limb_colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
                               [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255], [170, 170, 0],
                               [0, 170, 0], [255, 20, 0], [50, 0, 170], [0, 0, 255], [150, 170, 0],
                               [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255], [170, 170, 0],
                               [170, 0, 170]]
                if skeleton is not None:
                    skeleton = np.reshape(skeleton, (-1, 2))
                    neck = np.array((keypoints[i, 5, :] + keypoints[i, 6, :]) / 2).astype(int)
                    if keypoints[i, 5, 2] == 0 or keypoints[i, 6, 2] == 0:
                        neck = [0, 0, 0]
                    limb_index = -1
                    for limb in skeleton:
                        limb_index += 1
                        start_index, end_index = limb - 1  # connection stats from 1 to 17
                        if start_index == -1:
                            Joint_start = neck
                        else:
                            Joint_start = keypoints[i][start_index]
                        if end_index == -1:
                            Joint_end = neck
                        else:
                            Joint_end = keypoints[i][end_index]
                        # both are Annotated
                        # Joint:(x,y,v)
                        if (Joint_start[2] != 0) & (Joint_end[2] != 0):
                            cv2.line(image, tuple(Joint_start[:2]), tuple(Joint_end[:2]), limb_colors[limb_index], 5)

                mask = masks[:, :, i]
                image = visualize.apply_mask(image, mask, color)
                caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
                cv2.putText(image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color)
            return image

        # cap = cv2.VideoCapture(0) # uses the computer connected camera
        cap = cv2.VideoCapture(VIDEO_PATH)

        # Define the codec and create VideoWriter object.The output is stored in '<filename>.avi' file.
        # Define the fps to be equal to 10. Also frame size is passed.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(VIDEO_PATH_OUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame_width, frame_height))
        while True:
            ret, frame = cap.read()

            if ret:
                # get a frame
                ret, frame = cap.read()
                "BGR->RGB"
                if frame is None:
                    break
                rgb_frame = frame[:, :, ::-1]
                print(np.shape(frame))
                # Run detection
                t = time.time()
                results = model.detect([rgb_frame], verbose=0)
                # show a frame
                t = time.time() - t
                print(1.0 / t)
                r = results[0]  # for one image

                result_image = _cv2_display_keypoint(frame,
                                                     r['bboxes'],
                                                     r['keypoints'],
                                                     r['masks'],
                                                     r['class_ids'],
                                                     r['scores'],
                                                     class_names)

                out.write(result_image)
                # cv2.imshow('Detect image', result_image)
                # Press Q on keyboard to stop recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
