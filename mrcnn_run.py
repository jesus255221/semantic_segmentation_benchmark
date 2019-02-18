# Import configuration
import config as cfg
# Import mrcnn
from mrcnn import model as mrcnn_model
# Import utilities
import utils
# Import data generator and dataset
import data

import keras
import numpy as np
import random
import json

import argparse
# Import COCO tools
from pycocotools.coco import COCO
from pycocotools.cocostuffeval import COCOStuffeval
from pycocotools import cocostuffhelper
import keras.optimizers as KO
from keras.applications.resnet50 import preprocess_input
import keras.callbacks as KC
import keras.layers as KL
import keras.models as KM
from glob import glob
from tqdm import trange



def main():
    """main function

    Main function... (what do you expect me to say...)

    Args:
        - none

    Returns:
        - none
    """
    
    # Main function for evaluate
    parser = argparse.ArgumentParser(description = "A testing framework for MRCNN.")
    parser.add_argument("--epochs", required=False, default=500, type=int)
    parser.add_argument("--batch_size", required=False, default=16, type=int)
    parser.add_argument("--gpu_id", required=False, default="0", type=str, help="(str) The id of the gpu used when training.")
    parser.add_argument("--val_type", required=False, default="bbox", type=str, help="(str) Evaluation mode between segmentation and bounding box")
    
    
    # Parse argument
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    gpu_number = args.gpu_id
    
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
            
            
    Config = cfg.CocoConfig()
    Config.display()
    model = mrcnn_model.MaskRCNN(mode="training", config=Config,
                                  model_dir=Config.DEFAULT_LOGS_DIR)

    # Select weights file to load
    model_path = Config.COCO_MODEL_PATH

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = data.CocoDataset()
    dataset_train.load_coco(Config.COCO_training_ann_path, "train", year=Config.DEFAULT_DATASET_YEAR)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = data.CocoDataset()
    dataset_val.load_coco(Config.COCO_validation_ann_path, "val", year=Config.DEFAULT_DATASET_YEAR)
    dataset_val.prepare()
        
    print("Training network heads")
    imgaug = data.CocoDataset().augmentation
    model.train(dataset_train, dataset_val,
                learning_rate=Config.LEARNING_RATE,
                #epochs=40,
                epochs=1,
                layers='heads',
                augmentation=imgaug)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=Config.LEARNING_RATE,
                #epochs=120,
                epochs=1,
                layers='4+',
                augmentation=imgaug)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=Config.LEARNING_RATE / 10,
                #epochs=160,
                epochs=1,
                layers='all',
                    augmentation=imgaug)

    class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
        
    print("Evaluation Phase...")

    config = InferenceConfig()
    config.display()

    model = mrcnn_model.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    model_path = COCO_MODEL_PATH
    # Load weights
    print("Loading weights...", model_path)
    model.load_weights(model_path, by_name=True)
    # Validation dataset
    dataset_val = data.CocoDataset()
    coco = dataset_val.load_coco(Config.COCO_validation_ann_path, "val", year="2017", return_coco=True)
    dataset_val.prepare()
    print("Running COCO evaluation on 5000 images.")
    data.evaluate_coco(model, dataset_val, coco, "bbox", limit=5000)
    
if __name__ == '__main__':
    main()