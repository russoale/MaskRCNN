"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import sys
import time
import urllib.request
import zipfile

import numpy as np
import os
import shutil
from pycocotools import mask as maskUtils
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Root directory to hdd
DATA_DIR = os.path.abspath("/data/hdd/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import dataset
from mrcnn.load_weights import load_weights
from mrcnn.augmenter import FliplrKeypoint

# Local path to trained weights file

# mask only
COCO_MODEL_PATH_MASK = os.path.join(DATA_DIR, "russales", "logs", "coco20190402T2205", "mask_rcnn_coco_0160.h5")

# keypoint only
COCO_MODEL_PATH_KEYPOINT = os.path.join(DATA_DIR, "russales", "logs", "coco20190404T2343", "mask_rcnn_coco_0160.h5")

# keypoint and mask
COCO_MODEL_PATH = os.path.join(DATA_DIR, "russales", "mask_rcnn_coco_0160.h5")

# path to MSCOCO dataset
COCO_DIR = os.path.join(DATA_DIR, "Datasets", "MSCOCO_2017")

# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(DATA_DIR, "russales", "logs")

# Dataset default year, if not provided through the command line argument --logs
DEFAULT_DATASET_YEAR = "2017"


############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Person and background

    BACKBONE = "resnet50"

    NUM_KEYPOINTS = 17

    MASK_SHAPE = [28, 28]

    KEYPOINT_MASK_SHAPE = [56, 56]

    TRAIN_ROIS_PER_IMAGE = 100

    MAX_GT_INSTANCES = 128

    RPN_TRAIN_ANCHORS_PER_IMAGE = 150

    USE_MINI_MASK = True

    MASK_POOL_SIZE = 14

    KEYPOINT_MASK_POOL_SIZE = 7

    LEARNING_RATE = 0.001

    STEPS_PER_EPOCH = 1000

    KEYPOINT_LOSS_WEIGHTING = True

    KEYPOINT_THRESHOLD = 0.005

    PART_STR = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

    LIMBS = [0, -1, -1, 5, -1, 6, 5, 7, 6, 8, 7, 9, 8, 10, 11, 13, 12, 14, 13, 15, 14, 16]

    TRAINING_HEADS = None


############################################################
#  Dataset
############################################################

class CocoDataset(dataset.Dataset):

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def keypoint_names(self):
        return self._keypoint_names

    def __init__(self, task_type="instances"):
        assert task_type in ["instances", "person_keypoints"]
        self.task_type = task_type
        # the connection between 2 close keypoints
        self._skeleton = []
        # keypoint names
        # ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
        # "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        # "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        self._keypoint_names = []
        super().__init__()

    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/{}_{}{}.json".format(dataset_dir, self.task_type, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        if self.task_type == "person_keypoints":
            # the connection between 2 close keypoints
            self._skeleton = coco.loadCats(1)[0]["skeleton"]
            # keypoint names
            # ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
            # "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
            # "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
            self._keypoint_names = coco.loadCats(1)[0]["keypoints"]

            self._skeleton = np.array(self._skeleton, dtype=np.int32)

            print("Skeleton:", np.shape(self._skeleton))
            print("Keypoint names:", np.shape(self._keypoint_names))

        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_keypoints_mask(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        keypoints = []
        class_ids = []
        instance_masks = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            # only class person for keypoints
            assert class_id == 1

            if class_id:
                # load masks
                m = self.ann_to_mask(annotation, image_info["height"],
                                     image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                # load keypoints
                keypoint = annotation["keypoints"]
                keypoint = np.reshape(keypoint, (-1, 3))

                keypoints.append(keypoint)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            keypoints = np.array(keypoints, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            masks = np.stack(instance_masks, axis=2)
            return keypoints, masks, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_keypoints_mask(image_id)

    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            # only class person for keypoints
            assert class_id == 1

            if class_id:
                # load masks
                m = self.ann_to_mask(annotation, image_info["height"],
                                     image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                # load keypoints
                keypoint = annotation["keypoints"]
                keypoint = np.reshape(keypoint, (-1, 3))

                keypoints.append(keypoint)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            keypoints = np.array(keypoints, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return keypoints, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_keypoints_mask(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.ann_to_mask(annotation, image_info["height"],
                                     image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def load_bbox(self, image_id):
        """Load instance bbox for the given image.

        Different datasets use different ways to store bboxes. This
        function converts the different bbox format to one format
        in the form of array with coordinates in (y1,x1,y2,x2).

        Returns:
        bbox: Box in (y1,x1,y2,x2) format.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        annotations = image_info["annotations"]
        boxes = np.zeros([len(annotations), 4], dtype=np.int32)
        for i, annotation in enumerate(annotations):
            y1 = annotation["bbox"][1]  # y1
            x1 = annotation["bbox"][0]  # x1
            y2 = annotation["bbox"][3] + y1  # y2
            x2 = annotation["bbox"][2] + x1  # x2
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def ann_to_rle(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def ann_to_mask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, result):
    # Arrange resutls to match COCO specs in http://cocodataset.org/#format
    # If no results, return an empty list

    if result["bboxes"] is None:
        return []

    rois = result["bboxes"]
    class_ids = result["class_ids"]
    scores = result["scores"]
    masks = None
    keypoints = None

    if "masks" in result:
        masks = result["masks"].astype(np.uint8)

    if "keypoints" in result:
        keypoints = result["keypoints"]

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score
            }

            if masks is not None:
                mask = masks[:, :, i]
                result["segmentation"] = maskUtils.encode(np.asfortranarray(mask))
            if keypoints is not None:
                keypoint = keypoints[i, :, :].flatten().tolist()
                result["keypoints"] = keypoint

            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None, training_heads=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: 'segm', 'bbox', 'keypoints' for evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    total_image_count = len(image_ids)
    next_prgress = 0
    for i, image_id in enumerate(image_ids):
        # print progress
        progress = int(100 * (i / total_image_count))
        if next_prgress != progress:
            next_prgress = progress
            print('\r[{0}{1}] {2}%'.format('#' * progress, " " * (100 - progress), progress), end=' ', flush=True)

        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        if training_heads == "keypoint":
            r = model.detect_keypoint([image], verbose=0)[0]
        elif training_heads == "mask":
            r = model.detect_mask([image], verbose=0)[0]
        else:
            r = model.detect([image], verbose=0)[0]

        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1], r)
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default=COCO_DIR,
                        metavar="/path/to/coco/",
                        help="Directory of the MS-COCO dataset (default='/data/hdd/Datasets/MSCOCO_2017)'")
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2017)')
    parser.add_argument('--model', required=False,
                        default=COCO_MODEL_PATH,
                        metavar="<'/path/to/weights.h5'|'keypoint'|'mask'|'imagenet'>",
                        help="Path to weights .h5 file (default = keypoint & mask weights)")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    parser.add_argument('--training_heads', required=False,
                        default=None,
                        metavar="<'keypoint'|'mask'>",
                        help='Specify which head networks to train or evaluate (default=None, trains all)')
    parser.add_argument('--keypoint', required=False,
                        default=True,
                        metavar="<True|False>",
                        help='Include Keypoint Detection (default=True)',
                        type=bool)
    parser.add_argument('--continue_training', required=False,
                        default=False,
                        metavar="<True|False>",
                        help="Try to continue a previously started training, \
                                mainly by trying to recreate the optimizer state and epoch number.")
    parser.add_argument('--eval_type', required=False,
                        default="bbox",
                        metavar="<'segm'|'bbox'|'keypoints'>",
                        help="Set the type of official coco evaluation")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)
    print("Training Heads: ", args.training_heads)
    print("Keypoint: ", args.keypoint)
    print("Continue Training: ", args.continue_training)
    print("Limit: ", args.limit)
    print("Eval Type: ", args.eval_type)

    # Select weights file to load
    model_path = args.model
    if args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = utils.get_imagenet_weights()

    # select task type
    task_type = "person_keypoints" if args.keypoint else "instaces"

    # Train or evaluate
    if args.command == "train":
        config = CocoConfig()
        config.display()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

        # Training Heads
        config.TRAINING_HEADS = args.training_heads

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset(task_type=task_type)
        dataset_train.load_coco(args.dataset, "train", year=args.year, class_ids=[1], auto_download=args.download)
        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, class_ids=[1],
                                    auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset(task_type=task_type)
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, class_ids=[1], auto_download=args.download)
        dataset_val.prepare()

        print("Train Keypoints Image Count: {}".format(len(dataset_train.image_ids)))
        print("Train Keypoints Class Count: {}".format(dataset_train.num_classes))
        for i, info in enumerate(dataset_train.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        print("Val Keypoints Image Count: {}".format(len(dataset_val.image_ids)))
        print("Val Keypoints Class Count: {}".format(dataset_val.num_classes))
        for i, info in enumerate(dataset_val.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = FliplrKeypoint(0.5, config=config)

        # training phase schedule
        lr_values = [config.LEARNING_RATE * 2,
                     config.LEARNING_RATE,
                     config.LEARNING_RATE / 10]
        epochs_values = [40,
                         120,
                         160]
        trainable_layers = ["heads",
                            "4+",
                            "all"]

        last_layers = None
        last_epoch = None
        if model_path is not None and args.continue_training:
            last_epoch, _ = utils.get_epoch_and_date_from_model_path(model_path=model_path)
        weights_loaded = False
        can_load_optimizer_weights = args.continue_training

        # Run all training phases
        for i_t, (lr, epochs, layers) in enumerate(zip(lr_values, epochs_values, trainable_layers)):
            if last_epoch is not None and last_epoch >= epochs:
                # If we start with a new training phase, optimizer state can only be reused,
                # if the trainable layers did not change.
                if last_epoch == epochs:
                    if len(trainable_layers) > (i_t + 1) and layers != trainable_layers[i_t + 1]:
                        can_load_optimizer_weights = False
                # Skip this training stage as it has already been completed
                continue

            # If the trainable layers changed, we need to recompile the model
            if layers != last_layers:
                model.compile(layers=layers)

            # For the first training phase to run, load the initial weights
            if not weights_loaded and model_path is not None:
                # Load weights
                print("Loading weights from: ", model_path)
                load_weights(model, model_path, by_name=True,
                             exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
                             include_optimizer=can_load_optimizer_weights)
                weights_loaded = True

            print("Training: {}".format(layers))
            model.train(dataset_train, dataset_val, learning_rate=lr, epochs=epochs, augmentation=augmentation)

            last_layers = layers

    elif args.command == "evaluate":
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()
        # Training Heads
        config.TRAINING_HEADS = args.training_heads
        config.display()

        # Validation dataset
        dataset_val = CocoDataset(task_type=task_type)
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, class_ids=[1], return_coco=True,
                                     auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit if int(args.limit) != 0 else "all"))

        # Create model in inference mode
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

        # Load weights
        print("Loading weights from ", model_path)
        load_weights(model, model_path, by_name=True, include_optimizer=False)

        print("Start evaluation..")
        evaluate_coco(model, dataset_val, coco, args.eval_type, limit=int(args.limit),
                      training_heads=args.training_heads)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
