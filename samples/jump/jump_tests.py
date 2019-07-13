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
from imgaug import augmenters as iaa

import mrcnn.model as modellib
from mrcnn import augmenter
from mrcnn import data_generator, visualize
from mrcnn.load_weights import load_weights
from samples.jump import jump
from samples.jump.bisp_joint_order import JumpJointOrder

DATA_DIR = os.path.abspath("/data/hdd")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(DATA_DIR, "russales", "test_logs")

# Local path to trained weights file
JUMP_MODEL_PATH = os.path.join(DATA_DIR, "russales", "logs", "jump20190625T1622", "mask_rcnn_jump_0160.h5")

# Example Video Path
VIDEO_PATH = os.path.join(DATA_DIR, "russales", "JumpDataset", "Videos", "Dreisprung", "drei 180617 Garritsen 1.mp4")
VIDEO_PATH_OUT = os.path.join(DATA_DIR, "russales", "drei_180617_Garritsen_1_out_160.avi")


class JumpTests(TestCase):

    def load_test_config(self, training_heads):
        class TestConfig(jump.JumpConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            TRAINING_HEADS = training_heads

        config = TestConfig()
        config.display()
        return config

    def load_dataset(self, set="val"):
        # Validation dataset
        JUMP_DIR = os.path.abspath("/data/hdd/russales/JumpDataset/mscoco_format")
        dataset = jump.JumpDataset()
        dataset.load_jump(JUMP_DIR, "%s" % set)
        dataset.prepare()
        print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
        return dataset

    def test_load_mask(self):
        # load image and ground truth data
        config = self.load_test_config("mask")
        dataset = self.load_dataset()
        dg = data_generator.DataGenerator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)

        # image_id = random.choice(dataset.image_ids)
        image_id = 6

        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        augmentation = augmenter.FliplrKeypoint(1, config=config, dataset=dataset)

        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_keypoints, gt_mask_train = \
            dg.load_image_gt(image_id, augmentation=augmentation, use_mini_mask=False)

        # visualize.display_keypoints(image, gt_boxes, gt_keypoints, gt_class_ids, dataset.class_names,
        #                             ax=ax, title="Original", dataset=dataset)

        visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids,
                                    dataset.class_names, title="Original")
        pass

    def test_mask_augmentation(self):
        # load image and ground truth data
        config = self.load_test_config(None)
        dataset = self.load_dataset("train")
        dg = data_generator.DataGenerator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)

        image_id = random.choice(dataset.image_ids)
        # image_id = 2666
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))

        augmentation = iaa.Sequential([
            augmenter.FliplrKeypoint(0.5, config=config, dataset=dataset),
            iaa.Crop(percent=(0, 0.2)),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.ContrastNormalization((0.75, 1.5)),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            ),
            iaa.AssertShape((None, 1024, 1024, 3))
        ], random_order=True)

        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_keypoints, gt_mask_train = \
            dg.load_image_gt(image_id, augmentation=None, use_mini_mask=True)

        # if gt_masks.max() > 0:
        #     gt_masks = utils.expand_mask(gt_boxes, gt_masks, image.shape)
        #     visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids,
        #                                 dataset.class_names, title="Original")
        # else:

        jump_joint = JumpJointOrder()
        skeleton = np.array(jump_joint.bodypart_indices(), dtype=np.int32)
        for connection in skeleton:
            connection[0], connection[1] = connection + 1
        print(skeleton)
        visualize.display_keypoints(image, gt_boxes, gt_keypoints, gt_class_ids, dataset.class_names, skeleton=skeleton,
                                    title="Original", dataset=dataset)
        pass

    def test_predict_mask(self):
        # load image and ground truth data
        config = self.load_test_config(None)
        dataset = self.load_dataset("train")

        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

        # Load weights
        print("Loading weights ", JUMP_MODEL_PATH)
        load_weights(model, JUMP_MODEL_PATH, by_name=True, include_optimizer=False)

        dg = data_generator.DataGenerator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)

        image_ids = [id for id, img in dataset.jump.imgs.items() if "091217 Nogueira 6" in img['file_name']]

        # has_ann = []
        # for id in image_ids:
        #     _, _, mask_train = dataset.load_mask(id)
        #     has_ann.append(mask_train)

        # image_id = random.choice(dataset.image_ids)
        # image_ids = []
        # random.shuffle(dataset.image_ids)
        # for id in dataset.image_ids:
        #     _, _, mask_train = dataset.load_mask(id)
        #     if mask_train == 1:
        #         image_ids.append(id)
        #     if len(image_ids) == 5:
        #         break

        rows = math.ceil(len(image_ids) / 5)
        height = (len(image_ids) / 5) * 16
        f = plt.figure(figsize=(80, height))
        for idx, image_id in enumerate(image_ids):
            info = [i for i in dataset.image_info if i["id"] == image_id][0]
            _, _, mask_train = dataset.load_mask(dataset.image_info.index(info))
            print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                                   dataset.image_reference(image_id)))

            image_id = dataset.image_info.index([i for i in dataset.image_info if i["id"] == image_id][0])
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_keypoints, gt_mask_train = \
                dg.load_image_gt(image_id, augmentation=None, use_mini_mask=False)

            # create skeleton to display joints
            from samples.jump.bisp_joint_order import JumpJointOrder
            jump_joint = JumpJointOrder()
            skeleton = np.array(jump_joint.bodypart_indices(), dtype=np.int32)
            # print(skeleton)
            # only jump dataset related
            for connection in skeleton:
                connection[0], connection[1] = connection + 1

            results = model.detect([image], verbose=1)

            # Display results
            r = results[0]

            bboxes_res = r['bboxes']
            class_ids_res = r['class_ids']
            scores_res = r['scores']
            keypoints_res = r['keypoints']
            mask_res = r['masks']
            print("Mask detected: ", mask_res.max())
            ax = f.add_subplot(rows, 5, idx + 1)
            visualize.display_instances(image, bboxes_res, mask_res, class_ids_res, dataset.class_names, ax=ax,
                                        title="Mask annotation available: {}".format(mask_train))

        # ax0 = f.add_subplot(221)
        # ax1 = f.add_subplot(222)
        # ax2 = f.add_subplot(223)
        # ax3 = f.add_subplot(224)
        # visualize.display_keypoints(image, gt_boxes, gt_keypoints, gt_class_ids, dataset.class_names, skeleton=skeleton,
        #                             ax=ax0, title="Original", dataset=dataset)
        # visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids, dataset.class_names, ax=ax1,
        #                             title="Original")
        # visualize.display_keypoints(image, bboxes_res, keypoints_res, class_ids_res, dataset.class_names,
        #                             skeleton=skeleton, scores=scores_res,
        #                             title="Predictions", ax=ax2, dataset=dataset)
        # visualize.display_instances(image, bboxes_res, mask_res, class_ids_res, dataset.class_names,
        #                             title="Predictions", ax=ax3)

        # plt.show()
        # plt.savefig("jump20190625T1622/160/val/{}-{}.png".format(info["source"], info["id"]))
        # plt.savefig("jump20190625T1622/25/val/multiple.png", bbox_inches='tight')
        plt.savefig("jump20190625T1622/vid_sequence_Nogueira_train_25.png", bbox_inches='tight')

    def test_video(self):
        # load image and ground truth data
        config = self.load_test_config(None)
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
        # load_weights(model, "/Users/alessandrorusso/Downloads/mask_rcnn_jump_0160.h5", by_name=True, include_optimizer=False)
        load_weights(model, JUMP_MODEL_PATH, by_name=True, include_optimizer=False)

        class_names = ['BG', 'person']
        jump_joint = JumpJointOrder()
        skeleton = np.array(jump_joint.bodypart_indices(), dtype=np.int32)

        def _cv2_display_keypoint(image, boxes, keypoints, masks, class_ids, scores, class_names,
                                  skeleton=skeleton):
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
                               [170, 0, 170]]
                if len(skeleton):
                    skeleton = np.reshape(skeleton, (-1, 2))
                    neck = np.array((keypoints[i, 5, :] + keypoints[i, 6, :]) / 2).astype(int)
                    if keypoints[i, 5, 2] == 0 or keypoints[i, 6, 2] == 0:
                        neck = [0, 0, 0]
                    limb_index = -1
                    for limb in skeleton:
                        limb_index += 1
                        start_index, end_index = limb  # connection joint index from 0 to 16
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
                            # print(color)
                            cv2.line(image, tuple(Joint_start[:2]), tuple(Joint_end[:2]), limb_colors[limb_index], 5)
                mask = masks[:, :, i]
                image = visualize.apply_mask(image, mask, color)
                caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
                cv2.putText(image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color)
            return image

        # cap = cv2.VideoCapture(0) # uses the computer connected camera
        # cap = cv2.VideoCapture("/Users/alessandrorusso/Downloads/test.mp4")
        # out = cv2.VideoWriter('/Users/alessandrorusso/Downloads/test_out.avi', -1, 20.0, (1080, 1920))

        cap = cv2.VideoCapture(VIDEO_PATH)

        # Define the codec and create VideoWriter object.The output is stored in '<filename>.avi' file.
        # Define the fps to be equal to 10. Also frame size is passed.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(VIDEO_PATH_OUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame_width, frame_height))
        while (True):
            ret, frame = cap.read()

            if ret == True:
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
