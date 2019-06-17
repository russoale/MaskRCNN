import itertools
import json
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import os
import re
from imgaug import augmenters as iaa
from numpy import pi
from pycocotools import mask as maskUtils

# Root directory of the project

ROOT_DIR = os.path.abspath("../../")

# Root directory to hdd
DATA_DIR = os.path.abspath("/data/hdd/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib, dataset, utils, augmenter
from mrcnn.load_weights import load_weights
from mrcnn.config import Config

# path to MSCOCO dataset
JUMP_DIR = os.path.join(DATA_DIR, "russales", "JumpDataset", "mscoco_format")
# JUMP_DIR_IMAGE = os.path.join(DATA_DIR, "russales", "JumpDataset", "mscoco_format")
# JUMP_DIR = os.path.join(DATA_DIR, "russales")
MASK_JUMP_DIR = os.path.join(DATA_DIR, "russales", "JumpDataset", "Segmentation_masks", "annotierte_daten_springer")

# keypoint and mask pretrained weights
MODEL_PATH = os.path.join(DATA_DIR, "russales", "mask_keypoint_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(DATA_DIR, "russales", "logs")

# Dataset default year, if not provided through the command line argument --logs
DEFAULT_DATASET_YEAR = "2017"


############################################################
#  Configurations
############################################################

class JumpConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """

    # Configuration name
    NAME = "jump"
    IMAGES_PER_GPU = 2
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Person and background

    BACKBONE = "resnet50"

    NUM_KEYPOINTS = 20
    KEYPOINT_MASK_SHAPE = [56, 56]
    KEYPOINT_MASK_POOL_SIZE = 14
    KEYPOINT_THRESHOLD = 0.005
    PART_STR = ['head', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand', 'l_shoulder', 'l_elbow', 'l_wrist',
                'l_hand', 'r_hip', 'r_knee', 'r_ankle', 'r_heel', 'r_toetip', 'l_hip', 'l_knee', 'l_ankle',
                'l_heel', 'l_toetip']
    # LIMBS = [0,-1,-1,5,-1,6,5,7,6,8,7,9,8,10,11,13,12,14,13,15,14,16]

    TRAIN_ROIS_PER_IMAGE = 30
    MAX_GT_INSTANCES = 128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 150
    USE_MINI_MASK = True
    DETECTION_MAX_INSTANCES = 25

    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 704
    VALIDATION_STEPS = 49


############################################################
#  Jump Class
############################################################

class JUMP:
    def __init__(self, annotation_file=None):
        """
        Constructor of helper class for reading and visualizing annotations.
        Analogous to "COCO".
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert (type(data) == np.ndarray)
        print(data.shape)
        assert (data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i, N))
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann


############################################################
#  Dataset
############################################################

class JumpDataset(dataset.Dataset):
    def __init__(self):
        # assert task_type in ["instances", "person_keypoints"]
        # the connection between 2 close keypoints
        self.jump = dict()
        self._skeleton = []
        self._keypoint_names = ["head", "neck",
                                "rsho", "relb", "rwri", "rhan",
                                "lsho", "lelb", "lwri", "lhan",
                                "rhip", "rkne", "rank", "rhee", "rtoe",
                                "lhip", "lkne", "lank", "lhee", "ltoe"]
        super().__init__()

    def load_jump(self, dataset_dir, subset, class_ids=None, return_jump=False):
        """Load a subset of the jump dataset.
        dataset_dir: The root directory of the jump dataset.
        subset: What to load (train, val, minival, valminusminival)
        return_jump: If True, returns the JUMP object.
        """

        jump = JUMP("{}/annotations/keypoints_{}.json".format(dataset_dir, subset))
        self.jump = jump
        image_dir = "{}/{}".format(dataset_dir, subset)
        # image_dir = "{}/{}".format(JUMP_DIR_IMAGE, subset)

        # # Add classes
        if not class_ids:
            class_ids = sorted(jump.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(jump.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(jump.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("jump", i, jump.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "jump", image_id=i,
                path=os.path.join(image_dir, jump.imgs[i]['file_name']),
                width=jump.imgs[i]["width"],
                height=jump.imgs[i]["height"],
                annotations=jump.loadAnns(jump.getAnnIds(imgIds=[i])))

        print("Skeleton:", np.shape(self._skeleton))
        print("Keypoint names:", np.shape(self._keypoint_names))
        if return_jump:
            return jump

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def keypoint_names(self):
        return self._keypoint_names

    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        class_ids: a 1D array of class IDs of the instance masks, here is always equal to [num_person, 1]
        """

        keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = 1
            assert class_id == 1
            if class_id:
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
            return super(JumpDataset, self).load_keypoints(image_id)

    def get_keypoints(self):
        keypoints = self.keypoint_names
        keypoint_flip_map = {
            'rsho': 'lsho',
            'relb': 'lelb',
            'rwri': 'lwri',
            'rhan': 'lhan',
            'rhip': 'lhip',
            'rkne': 'lkne',
            'rank': 'lank',
            'rhee': 'lhee',
            'rtoe': 'ltoe',
        }
        return keypoints, keypoint_flip_map

    def random_zoom(self, bbox, shape, min_size):
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[0][0] + bbox[0][2], bbox[0][1] + bbox[0][3]
        if x1 > 0:
            b_x = x1 - np.random.randint(x1)
        else:
            b_x = 0
        if y1 > 0:
            b_y = y1 - np.random.randint(y1)
        else:
            b_y = 0

        if shape[1] - b_x < min_size:
            b_x = np.random.randint(shape[1] - min_size)
        if shape[0] - b_y < min_size:
            b_y = np.random.randint(shape[0] - min_size)

        if shape[1] - x2 > 0:
            bb_x = x2 + np.random.randint(shape[1] - x2)
        else:
            bb_x = x2
        if shape[0] - y2 > 0:
            bb_y = y2 + np.random.randint(shape[0] - y2)
        else:
            bb_y = y2

        return [b_x, b_y, bb_x, bb_y]

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        mask_train: A 1D array with binary flag 0,1 to for mask loss
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "jump":
            return super(JumpDataset, self).load_mask(image_id)

        instance_masks = []
        train_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        # start_time = time.time()
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "jump.{}".format(annotation['category_id']))
            if class_id:
                # m = self.ann_to_mask(annotation, image_info["height"], image_info["width"])

                idx = annotation["image_id"]
                # print("Image Id: {}, length of images: {}".format(idx, len(self.jump.imgs)))
                # remove .jpeg file type
                file_name = self.jump.imgs[idx]['file_name'][:-5]
                # create directory pattern
                directory = re.sub("_", "_(0", file_name) + ")"
                # check if segmentation available
                directory = os.path.join(MASK_JUMP_DIR, directory)

                # print("checking if directory : {} exists".format(directory))
                m = None
                if os.path.isdir(directory):
                    m = self.pickle_to_mask(directory)

                # multiplier for mask loss calculation
                train = 0 if m is None else 1

                # Some images don't have any segmentation annotations.
                # To avoid skipping to many training examples generating empty mask
                # will still allow the network to train with but need to be excluded from
                # mask loss calculation.
                if m is None:
                    # image = self.jump.imgs[image_id]
                    # width = image["width"]
                    # height = image["height"]
                    # Annotations for pictures might be switched in some cases
                    # general all pictures in Jump dataset are of shape (WxHxC) 1920 x 1080 x 3
                    # image: [height, width, 3]
                    m = np.zeros((1080, 1920, 3)).astype(bool)

                instance_masks.append(m)
                train_masks.append(train)
                class_ids.append(class_id)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Pack instance masks into an array
        if class_ids:
            mask = np.squeeze(instance_masks)[:, :, :1]
            mask_train = np.array(train_masks, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids, mask_train
        else:
            # Call super class to return an empty mask
            return super(JumpDataset, self).load_mask(image_id)

    def ann_to_rle(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if 'segmentation' in ann:
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
        else:
            return None

    def ann_to_mask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann, height, width)
        m = None if rle is None else maskUtils.decode(rle)
        return m

    def pickle_to_mask(self, directory):
        """
        Search in directory for segmentation pickle and return if available
        :return: binary mask (numpy 2D array)
        """
        regex = re.compile('(.*whole_person.mask.pickle$)')
        for _, _, files in os.walk(directory):
            for file in files:
                if regex.match(file):
                    with open(os.path.join(directory, file), "rb") as pickle_file:
                        rle = pickle.load(pickle_file)
                    m = maskUtils.decode(rle)
                    return np.asarray(m).astype(np.bool)
        return None

    def get_bbox_from_keypoints(self, keypoints_list, image_shape):
        """
        Pls, don't ask ...
        :return: Box in (y1,x1,y2,x2) format.
        """
        eps = 1e-7

        num_person = keypoints_list.shape[0]
        boxes = np.zeros([num_person, 4], dtype=np.int32)
        for i in range(num_person):
            keypoints = keypoints_list[i, :, :]
            valid_keypoints = keypoints[:, 2] > 0
            x_min = np.min(keypoints[:, 0][valid_keypoints])
            y_min = np.min(keypoints[:, 1][valid_keypoints])

            x_max = np.max(keypoints[:, 0][valid_keypoints])
            y_max = np.max(keypoints[:, 1][valid_keypoints])

            headtip_x = handtip_rx = handtip_lx = elbowtip_rx = elbowtip_lx = kneetip_rx = kneetip_lx = heeltip_rx = \
                heeltip_lx = toetip_rx = toetip_lx = hiptip_lx = hiptip_rx = shouldertip_rx = shouldertip_lx = \
                ank_kneetip_rx = ank_kneetip_lx = x_min

            headtip_y = handtip_ry = handtip_ly = elbowtip_ry = elbowtip_ly = kneetip_ry = kneetip_ly = heeltip_ry = \
                heeltip_ly = toetip_ry = toetip_ly = hiptip_ly = hiptip_ry = shouldertip_ry = shouldertip_ly = \
                ank_kneetip_ry = ank_kneetip_ly = y_min

            # Head-Neck

            if keypoints[0][2] > 0 and keypoints[1][2] > 0:
                slope = np.float((keypoints[1][1] - keypoints[0][1]) / (keypoints[1][0] - keypoints[0][0] + eps))
                ang = np.abs(np.arctan(slope))
                if ang < pi / 24:
                    headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 11 * keypoints[0][1] - 10 * \
                                           keypoints[1][
                                               1]
                elif ang < pi / 12:
                    headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 7 * keypoints[0][1] - 6 * \
                                           keypoints[1][1]
                elif ang < pi / 6:
                    headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 5 * keypoints[0][1] - 4 * \
                                           keypoints[1][1]
                elif ang < pi / 4:
                    headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 3.5 * keypoints[0][1] - 2.5 * \
                                           keypoints[1][1]
                elif ang < pi / 3:
                    headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 3 * keypoints[0][1] - 2 * \
                                           keypoints[1][1]
                else:
                    headtip_x, headtip_y = 3 * keypoints[0][0] - 2 * keypoints[1][0], 2.6 * keypoints[0][1] - 1.6 * \
                                           keypoints[1][1]
            # elif keypoints[0][2] == 0:
            #     y_min = 0

            # Hands
            if keypoints[4][2] > 0 and keypoints[5][2] > 0:
                handtip_rx, handtip_ry = 3 * keypoints[5][0] - 2 * keypoints[4][0], 3 * keypoints[5][1] - 2 * \
                                         keypoints[4][
                                             1]
            # elif keypoints[5][2] == 0:
            #     y_min = 0

            if keypoints[8][2] > 0 and keypoints[9][2] > 0:
                handtip_lx, handtip_ly = 3 * keypoints[9][0] - 2 * keypoints[8][0], 3 * keypoints[9][1] - 2 * \
                                         keypoints[8][
                                             1]
            # elif keypoints[9][2] == 0:
            #     y_min = 0

            # Shoulder-elbow
            if keypoints[2][2] > 0 and keypoints[3][2] > 0:
                elbowtip_rx, elbowtip_ry = (10 * keypoints[3][0] - 3 * keypoints[2][0]) / 7.0, (
                        10 * keypoints[3][1] - 3 * keypoints[2][1]) / 7.0
                shouldertip_rx, shouldertip_ry = (10 * keypoints[2][0] - 3 * keypoints[3][0]) / 7.0, (
                        10 * keypoints[2][1] - 3 * keypoints[3][1]) / 7.0

            if keypoints[6][2] > 0 and keypoints[7][2] > 0:
                elbowtip_lx, elbowtip_ly = (10 * keypoints[7][0] - 3 * keypoints[6][0]) / 7.0, (
                        10 * keypoints[7][1] - 3 * keypoints[6][1]) / 7.0
                shouldertip_lx, shouldertip_ly = (10 * keypoints[6][0] - 3 * keypoints[7][0]) / 7.0, (
                        10 * keypoints[6][1] - 3 * keypoints[7][1]) / 7.0

            # Hip-Knee
            if keypoints[10][2] > 0 and keypoints[11][2] > 0:
                kneetip_rx, kneetip_ry = (10 * keypoints[11][0] - 2 * keypoints[10][0]) / 8.0, (
                        10 * keypoints[11][1] - 2 * keypoints[10][1]) / 8.0
                hiptip_rx, hiptip_ry = (10 * keypoints[10][0] - 3 * keypoints[11][0]) / 7.0, (
                        10 * keypoints[10][1] - 3 * keypoints[11][1]) / 7.0

            if keypoints[15][2] > 0 and keypoints[16][2] > 0:
                kneetip_lx, kneetip_ly = (10 * keypoints[16][0] - 2 * keypoints[15][0]) / 8.0, (
                        10 * keypoints[16][1] - 2 * keypoints[15][1]) / 8.0
                hiptip_lx, hiptip_ly = (10 * keypoints[15][0] - 3 * keypoints[16][0]) / 7.0, (
                        10 * keypoints[15][1] - 3 * keypoints[16][1]) / 7.0

            # Ankle-Knee
            if keypoints[11][2] > 0 and keypoints[12][2] > 0:
                ank_kneetip_rx, ank_kneetip_ry = (10 * keypoints[11][0] - 2.5 * keypoints[12][0]) / 7.5, (
                        10 * keypoints[11][1] - 2.5 * keypoints[12][1]) / 7.5

            if keypoints[16][2] > 0 and keypoints[17][2] > 0:
                ank_kneetip_lx, ank_kneetip_ly = (10 * keypoints[16][0] - 2.5 * keypoints[17][0]) / 7.5, (
                        10 * keypoints[16][1] - 2.5 * keypoints[17][1]) / 7.5

            # Ankle-Heel
            if keypoints[12][2] > 0 and keypoints[13][2] > 0:
                slope = np.float((keypoints[13][1] - keypoints[12][1]) / (keypoints[13][0] - keypoints[12][0] + eps))
                ang = np.abs(np.arctan(slope))
                if ang < pi / 4:
                    heeltip_rx, heeltip_ry = (2 * keypoints[13][0] - 1 * keypoints[12][0]), (
                            2.5 * keypoints[13][1] - 1.5 * keypoints[12][1])
                else:
                    heeltip_rx, heeltip_ry = (2.5 * keypoints[13][0] - 1.5 * keypoints[12][0]), (
                            2 * keypoints[13][1] - 1 * keypoints[12][1])

            if keypoints[17][2] > 0 and keypoints[18][2] > 0:
                slope = np.float((keypoints[18][1] - keypoints[17][1]) / (keypoints[18][0] - keypoints[17][0] + eps))
                ang = np.abs(np.arctan(slope))
                if ang < pi / 4:
                    heeltip_lx, heeltip_ly = (2 * keypoints[18][0] - keypoints[17][0]), (
                            2.5 * keypoints[18][1] - 1.5 * keypoints[17][1])
                else:
                    heeltip_lx, heeltip_ly = (2.5 * keypoints[18][0] - 1.5 * keypoints[17][0]), (
                            2 * keypoints[18][1] - keypoints[17][1])

            # Heel-Toe
            if keypoints[13][2] > 0 and keypoints[14][2] > 0:
                slope = np.float((keypoints[13][1] - keypoints[14][1]) / (keypoints[13][0] - keypoints[14][0] + eps))
                ang = np.abs(np.arctan(slope))
                if ang < pi / 4:
                    toetip_rx, toetip_ry = (10 * keypoints[14][0] - 2 * keypoints[13][0]) / 8.0, (
                            10 * keypoints[14][1] - 3 * keypoints[13][1]) / 7.0
                else:
                    toetip_rx, toetip_ry = (10 * keypoints[14][0] - 3 * keypoints[13][0]) / 7.0, (
                            10 * keypoints[14][1] - 2 * keypoints[13][1]) / 8.0

            if keypoints[18][2] > 0 and keypoints[19][2] > 0:
                slope = np.float((keypoints[18][1] - keypoints[19][1]) / (keypoints[18][0] - keypoints[19][0] + eps))
                ang = np.abs(np.arctan(slope))
                if ang < pi / 4:
                    toetip_lx, toetip_ly = (10 * keypoints[19][0] - 2 * keypoints[18][0]) / 8.0, (
                            10 * keypoints[19][1] - 3 * keypoints[18][1]) / 7.0
                else:
                    toetip_lx, toetip_ly = (10 * keypoints[19][0] - 3 * keypoints[18][0]) / 7.0, (
                            10 * keypoints[19][1] - 2 * keypoints[18][1]) / 8.0

            xs = np.array(
                [headtip_x, handtip_rx, handtip_lx, elbowtip_lx, elbowtip_rx, kneetip_rx, kneetip_lx, heeltip_rx,
                 heeltip_lx,
                 toetip_rx, toetip_lx, hiptip_lx, hiptip_rx, shouldertip_rx, shouldertip_lx, ank_kneetip_rx,
                 ank_kneetip_lx])
            #     x_min = min(x_min, np.min(xs[np.logical_and(xs>=0, xs<image_shape[1])]))
            #     x_max = max(x_max, np.max(xs[np.logical_and(xs>=0, xs<image_shape[1])]))
            x_min = max(0, min(x_min, np.min(xs)))
            x_max = min(image_shape[1], max(x_max, np.max(xs)))

            ys = np.array(
                [headtip_y, handtip_ry, handtip_ly, elbowtip_ly, elbowtip_ry, kneetip_ry, kneetip_ly, heeltip_ry,
                 heeltip_ly,
                 toetip_ry, toetip_ly, hiptip_ly, hiptip_ry, shouldertip_ry, shouldertip_ly, ank_kneetip_ry,
                 ank_kneetip_ly])
            #     y_min = min(y_min, np.min(ys[np.logical_and(ys>=0, ys<image_shape[0])]))
            #     y_max = max(y_max, np.max(ys[np.logical_and(ys>=0, ys<image_shape[0])]))
            y_min = max(0, min(y_min, np.min(ys)))
            y_max = min(image_shape[0], max(y_max, np.max(ys)))

            x_top, y_top, x_bot, y_bot = int(np.floor(x_min)), int(np.floor(y_min)), int(np.ceil(x_max)), int(
                np.ceil(y_max))
            boxes[i] = np.array([y_top, x_top, y_bot, x_bot])

        return boxes.astype(np.int32)


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
                        default=JUMP_DIR,
                        metavar="/path/to/coco/",
                        help="Directory of the MS-COCO dataset (default='/data/hdd/Datasets/MSCOCO_2017)'")
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2017)')
    parser.add_argument('--model', required=False,
                        default=MODEL_PATH,
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

    # Train or evaluate
    if args.command == "train":
        config = JumpConfig()

        # Training Heads
        config.TRAINING_HEADS = args.training_heads
        config.display()

        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = JumpDataset()
        dataset_train.load_jump(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = JumpDataset()
        dataset_val.load_jump(args.dataset, "val")
        dataset_val.prepare()

        print("Train Keypoints Image Count: {}".format(len(dataset_train.image_ids)))
        print("Train Keypoints Class Count: {}".format(dataset_train.num_classes))
        for i, info in enumerate(dataset_train.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        print("Val Keypoints Image Count: {}".format(len(dataset_val.image_ids)))
        print("Val Keypoints Class Count: {}".format(dataset_val.num_classes))
        for i, info in enumerate(dataset_val.class_info):
            print("{:3}. {:50}".format(i, info['name']))

        # ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        exclude = ["mrcnn_keypoint_mask_deconv"]

        # Image Augmentation
        augmentation = iaa.Sequential([
            augmenter.FliplrKeypoint(0.5, config=config, dataset=dataset_train),
            iaa.Crop(percent=(0, 0.2)),
            # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # iaa.ContrastNormalization((0.75, 1.5)),
            # iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                # shear=(-8, 8)
            ),
            # iaa.AssertShape((None, 1024, 1024, 3))
        ], random_order=True)

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
                             exclude=exclude,
                             include_optimizer=can_load_optimizer_weights)
                weights_loaded = True

            print("Training: {}".format(layers))
            model.train(dataset_train, dataset_val, learning_rate=lr, epochs=epochs, augmentation=augmentation)

            last_layers = layers

    elif args.command == "evaluate":
        print("'evaluate' is not supported yet ...")

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
