"""
Mask R-CNN
Train on the toy Target dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 target.py train --dataset=/path/to/target/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 target.py train --dataset=/path/to/target/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 target.py train --dataset=/path/to/target/dataset --weights=imagenet

    # Apply color splash to an image
    python3 target.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 target.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import imgaug

import json
import datetime
import numpy as np
import skimage.draw
import glob
import cv2
import time
from mrcnn import visualize
import json
from dotmap import DotMap

# # Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
#
# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"

config_path = "configs/custom_data.json"
with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

_config = DotMap(config_dict)

############################################################
#  Configurations
############################################################


class TargetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "custom"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = _config.train.model.batch_size

    ## Backbone Architecture
    BACKBONE = _config.train.model.backbone
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + @

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(TargetConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = _config.test.model.batch_size


############################################################
#  Dataset
############################################################

class TargetDataset(utils.Dataset):

    def load_target(self, dataset_dir, subset):
        """Load a subset of the Target dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("custom", 1, "defect")

        # Train or validation dataset?
        assert subset in ["train", "val", "test_crop"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                shape_attributes = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                shape_attributes = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.dataset_size = len(image)
            self.add_image(
                "custom",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                shape_attributes=shape_attributes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a target dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shape_attributes"])],
                        dtype=np.uint8)

        if _config.dataset.mask == 'polygon':
            for i, p in enumerate(info["shape_attributes"]):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
        elif _config.dataset.mask == 'rect':
            # rect ((x1, y1), width, height)
            for i, p in enumerate(info["shape_attributes"]):
                # Get indexes of pixels inside the polygon and set them to 1
                # p['x']+p['height'],p['y']+p['width'])
                rr, cc = skimage.draw.rectangle(start=(p['y'],p['x']), extent=(p['height'],p['width']), shape=mask.shape)
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TargetDataset()
    dataset_train.load_target(_config.dataset.root, _config.dataset.train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TargetDataset()
    dataset_val.load_target(_config.dataset.root, _config.dataset.val)
    dataset_val.prepare()

    # # *** This training schedule is an example. Update to your needs ***
    # # Since we're using a very small dataset, and starting from
    # # COCO trained weights, we don't need to train too long. Also,
    # # no need to train all layers, just the heads should do it.
    # print("Training network heads")

    '''
    # Normal Training
    '''
    augmentation = imgaug.augmenters.Fliplr(0.5)
    if _config.train.method == 'normal':
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=_config.train.model.epoch,
                    layers='heads',
                    augmentation=augmentation)
    elif _config.train.method == 'special':
        '''
        # Special Training
        '''
        ## Training - Config
        starting_epoch = _config.train.model.epoch
        epoch = dataset_train.dataset_size // (config.STEPS_PER_EPOCH * config.BATCH_SIZE)
        epochs_warmup = 1 * epoch
        epochs_heads = 7 * epoch  # + starting_epoch
        epochs_stage4 = 7 * epoch  # + starting_epoch
        epochs_all = 7 * epoch  # + starting_epoch
        epochs_breakOfDawn = 5 * epoch
        print("> Training Schedule: \
            \nwarmup: {} epochs \
            \nheads: {} epochs \
            \nstage4+: {} epochs \
            \nall layers: {} epochs \
            \ntill the break of Dawn: {} epochs".format(
            epochs_warmup, epochs_heads, epochs_stage4, epochs_all, epochs_breakOfDawn))

        ## Training - WarmUp Stage
        print("> Warm Up all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=epochs_warmup,
                    layers='all',
                    augmentation=augmentation)

        ## Training - Stage 1
        print("> Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs_warmup + epochs_heads,
                    layers='heads',
                    augmentation=augmentation)

        ## Training - Stage 2
        # Finetune layers  stage 4 and up
        print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=epochs_warmup + epochs_heads + epochs_stage4,
                    layers="4+",
                    augmentation=augmentation)

        ## Training - Stage 3
        # Fine tune all layers
        print("> Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=epochs_warmup + epochs_heads + epochs_stage4 + epochs_all,
                    layers='all',
                    augmentation=augmentation)

        ## Training - Stage 4
        # Fine tune all layers
        print("> Fine tune all layers till the break of Dawn")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 100,
                    epochs=epochs_warmup + epochs_heads + epochs_stage4 + epochs_all + epochs_breakOfDawn,
                    layers='all',
                    augmentation=augmentation)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def inference(model, config=None):

    test_dir = os.path.join(_config.dataset.root, _config.dataset.test)
    img_paths = glob.glob(os.path.join(test_dir, '*.' + _config.dataset.img_extension))
    img_list = [cv2.imread(img_path) for img_path in img_paths]

    save_dir = os.path.join(test_dir, 'inference_' + _config.test.model.backbone)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for i, img in enumerate(img_list):
        print(img_paths[i].split(test_dir+'/')[1])

    start_idx = 0
    while True:
        T_start = time.time()

        results = []
        if (start_idx+config.IMAGES_PER_GPU) > (len(img_list) - 1):
            end_idx = len(img_list)
        else:
            end_idx = start_idx+config.IMAGES_PER_GPU

        results = model.detect(img_list[start_idx:end_idx],
                               batch_size=end_idx - start_idx,
                               verbose=1)

        print(config.IMAGES_PER_GPU)
        print(len(img_list[0:config.IMAGES_PER_GPU]))


        for i, r in enumerate(results):
            visualize.display_results(img_list[start_idx + i], r['rois'], r['masks'], r['class_ids'], ['BG','Defect'], r['scores'],
                                      save_dir=save_dir, img_name=img_paths[start_idx + i].split(test_dir+'/')[1],
                                      display_img=False)

        print('time:', (time.time()-T_start), 'len: ', config.IMAGES_PER_GPU)
        print((time.time()-T_start) / config.IMAGES_PER_GPU )

        if end_idx == len(img_list) - 1:
            break
        start_idx += config.IMAGES_PER_GPU


# ############################################################
# #  Detection
# ############################################################
#
# def detect(model, dataset_dir, subset):
#     """Run detection on images in the given directory."""
#     print("Running on {}".format(dataset_dir))
#
#     # Create directory
#     if not os.path.exists(RESULTS_DIR):
#         os.makedirs(RESULTS_DIR)
#     submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
#     submit_dir = os.path.join(RESULTS_DIR, submit_dir)
#     os.makedirs(submit_dir)
#
#     # Read dataset
#     dataset = NucleusDataset()
#     dataset.load_nucleus(dataset_dir, subset)
#     dataset.prepare()
#     # Load over images
#     submission = []
#     for image_id in dataset.image_ids:
#         # Load image and run detection
#         image = dataset.load_image(image_id)
#         # Detect objects
#         r = model.detect([image], verbose=0)[0]
#         # Encode image to RLE. Returns a string of multiple lines
#         source_id = dataset.image_info[image_id]["id"]
#         rle = mask_to_rle(source_id, r["masks"], r["scores"])
#         submission.append(rle)
#         # Save image with masks
#         visualize.display_instances(
#             image, r['rois'], r['masks'], r['class_ids'],
#             dataset.class_names, r['scores'],
#             show_bbox=False, show_mask=False,
#             title="Predictions")
#         plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))
#
#     # Save to csv file
#     submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
#     file_path = os.path.join(submit_dir, "submit.csv")
#     with open(file_path, "w") as f:
#         f.write(submission)
#     print("Saved to ", submit_dir)


############################################################
#  Command Line
############################################################


if __name__ == '__main__':
    import argparse

    # train
    # python3 target.py train --dataset="/home/dl8/Desktop/workspace/khj/Mask_RCNN-master/datasets/test" --weights="coco"
    # python3 target.py train --dataset=/path/to/dataset --model=last

    # # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN to detect targets.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'inference'")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/target/dataset/",
    #                     help='Directory of the Target dataset')
    # parser.add_argument('--weights', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--image', required=False,
    #                     metavar="path or URL to image",
    #                     help='Image to apply the color splash effect on')
    # parser.add_argument('--video', required=False,
    #                     metavar="path or URL to video",
    #                     help='Video to apply the color splash effect on')
    # parser.add_argument('--subset', required=False,
    #                     metavar="Dataset sub-directory",
    #                     help="Subset of dataset to run prediction on")
    #
    # args = parser.parse_args()
    # test --dataset="/home/dl8/Desktop/workspace/khj/Mask_RCNN-master/datasets/custom" --image "/home/dl8/Desktop/workspace/khj/Mask_RCNN-master/datasets/custom/test_crop/2_[1000 2021]_3.png" --weights=""/home/dl8/Desktop/workspace/khj/Mask_RCNN-master/logs/custom20191105T2159/mask_rcnn_custom_0018.h5"






    if _config.exp.mode == "train":
        print("Weights: ", _config.train.model.pre_weight)
        print("save model path: ", _config.train.model.save_path)

        config = TargetConfig()
        config.display()
        # Create model
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=_config.train.model.save_path)
        # Load weights
        # Select weights file to load
        if _config.train.model.pre_weight == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif _config.train.model.pre_weight == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif _config.train.model.pre_weight == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = _config.train.model.pre_weight

        print("Loading weights ", weights_path)

        if _config.train.model.pre_weight == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

        # train
        train(model)
    elif _config.exp.mode == "test":
        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=_config.train.model.save_path)

        # Load weights
        weights_path = _config.test.model.weight_path
        print("Loading weights ", weights_path)

        model.load_weights(weights_path, by_name=True)

        inference(model, config=config)