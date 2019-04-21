############################################################
#  Test
############################################################
from unittest import TestCase

import os
import random

from mrcnn import augmenter
from mrcnn import data_generator, visualize
from samples.jump import jump
from imgaug import augmenters as iaa


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

    def load_dataset(self):
        # Validation dataset
        JUMP_DIR = os.path.abspath("/data/hdd/russales/JumpDataset/mscoco_format")
        dataset = jump.JumpDataset()
        dataset.load_jump(JUMP_DIR, "val")
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
        dataset = self.load_dataset()
        dg = data_generator.DataGenerator(dataset, config, shuffle=True, batch_size=config.BATCH_SIZE)

        # image_id = random.choice(dataset.image_ids)
        image_id = 88
        image_id = 6

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
            dg.load_image_gt(image_id, augmentation=augmentation, use_mini_mask=False)

        #visualize.display_keypoints(image, gt_boxes, gt_keypoints, gt_class_ids, dataset.class_names,
        #                            title="Original", dataset=dataset)

        visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids,
                                     dataset.class_names, title="Original")
        pass
