############################################################
#  Test
############################################################
import json
import pickle
from unittest import TestCase

import copy
import numpy as np
import os
import random
import re
from PIL import Image, ImageDraw
from imgaug import augmenters as iaa
from pycocotools import mask as maskUtils
from skimage import measure

from mrcnn import augmenter, utils
from mrcnn import data_generator, visualize
from samples.jump import jump


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
        from samples.jump.bisp_joint_order import JumpJointOrder
        jump_joint = JumpJointOrder()
        skeleton = np.array(jump_joint.bodypart_indices(), dtype=np.int32)
        for connection in skeleton:
            connection[0], connection[1] = connection + 1
        print(skeleton)
        visualize.display_keypoints(image, gt_boxes, gt_keypoints, gt_class_ids, dataset.class_names, skeleton=skeleton,
                                        title="Original", dataset=dataset)
        pass

    def test_mask_into_annotation_json(self):
        DATA_DIR = os.path.abspath("/data/hdd/")
        JUMP_DIR = os.path.join(DATA_DIR, "russales", "JumpDataset", "mscoco_format")
        MASK_JUMP_DIR = os.path.join(DATA_DIR, "russales", "JumpDataset", "Segmentation_masks",
                                     "annotierte_daten_springer")

        json_file = json.load(open("{}/annotations/keypoints_{}.json".format(JUMP_DIR, "val"), 'r'))
        copy_file = copy.deepcopy(json_file)

        for image in json_file['images']:
            image_id = image['id']
            file_name = image['file_name'][:-5]
            directory = re.sub("_", "_(0", file_name) + ")"
            directory = os.path.join(MASK_JUMP_DIR, directory)

            if os.path.isdir(directory):
                regex = re.compile('(.*whole_person.mask.pickle$)')
                for _, _, files in os.walk(directory):
                    for file in files:
                        if regex.match(file):
                            with open(os.path.join(directory, file), "rb") as pickle_file:
                                rle = pickle.load(pickle_file)
                                m = maskUtils.decode(rle)
                                fortran_ground_truth_binary_mask = np.asfortranarray(m)
                                encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
                                ground_truth_area = maskUtils.area(encoded_ground_truth)

                                m = np.asarray(m)[:, :, :1]
                                m = m.reshape(m.shape[0], (m.shape[1] * m.shape[2]))
                                contours = measure.find_contours(m.astype(np.int8), 0.5)

                                for item in copy_file["annotations"]:
                                    if item["image_id"] == image_id:
                                        ann = item
                                        break
                                    else:
                                        continue

                                ann["segmentation"] = []
                                for contour in contours:
                                    contour = np.flip(contour, axis=1)
                                    segmentation = contour.ravel().tolist()

                                ann["segmentation"].append(segmentation)
                                ann["area"] = ground_truth_area.tolist()

        with open("/data/hdd/russales/annotations/keypoints_{}.json".format("val"), 'w') as fp:
            json.dump(copy_file, fp)

    def test_polygon_to_image(self):

        DATA_DIR = os.path.abspath("/data/hdd/")
        JUMP_DIR = os.path.join(DATA_DIR, "russales", "JumpDataset", "mscoco_format")
        MASK_JUMP_DIR = os.path.join(DATA_DIR, "russales", "JumpDataset", "Segmentation_masks",
                                     "annotierte_daten_springer")

        json_file = json.load(open("{}/annotations/keypoints_{}.json".format(JUMP_DIR, "train"), 'r'))

        copy_file = copy.deepcopy(json_file)

        for image in json_file['images']:
            image_id = image['id']
            file_name = image['file_name'][:-5]
            directory = re.sub("_", "_(0", file_name) + ")"
            directory = os.path.join(MASK_JUMP_DIR, directory)

            if os.path.isdir(directory):
                regex = re.compile('(.*whole_person.mask.pickle$)')
                for _, _, files in os.walk(directory):
                    for file in files:
                        if regex.match(file):
                            with open(os.path.join(directory, file), "rb") as pickle_file:
                                rle = pickle.load(pickle_file)
                                m = maskUtils.decode(rle)
                                fortran_ground_truth_binary_mask = np.asfortranarray(m)
                                encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
                                ground_truth_area = maskUtils.area(encoded_ground_truth)

                                m = np.asarray(m)[:, :, :1]
                                m = m.reshape(m.shape[0], (m.shape[1] * m.shape[2]))
                                contours = measure.find_contours(m.astype(np.int8), 0.5)

                                ann = next(item for item in copy_file["annotations"] if item["id"] == image_id)
                                ann["segmentation"] = []
                                for contour in contours:
                                    contour = np.flip(contour, axis=1)
                                    img = Image.new('L', (m.shape[0], m.shape[1]), 0)
                                    ImageDraw.Draw(img).polygon(contour, outline=1, fill=1)
                                    mask = np.array(img)

    def test_polygon(self):
        import json
        import numpy as np
        from pycocotools import mask
        from skimage import measure

        ground_truth_binary_mask = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)

        annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": 123,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": 1,
            "id": 1
        }

        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)

        print(json.dumps(annotation, indent=4))
