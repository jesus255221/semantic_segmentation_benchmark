# Import configuration
import config as cfg
# Import Unet and DeepLab
import basic_model
# Import mrcnn
from mrcnn import model as mrcnn_model
# Import custom model
import model
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
    parser = argparse.ArgumentParser(description = "A testing framework for detection, semantic seg and instance seg.")
    parser.add_argument("--net", help="The type of net work which is either mrcnn, unet, deeplab or custom.",
                       required=True, default="unet")
    parser.add_argument("--epochs", required=False, default=500, type=int)
    parser.add_argument("--batch_size", required=False, default=16, type=int)
    parser.add_argument("--gpu", required=False, default="0", type=str, help="The id of the gpu used when training.")
    parser.add_argument("--img_size", required=False, default=192, type=int, help="The size of input image")
    parser.add_argument("--load_weights", required=False, default=False, type=bool, help="Use old weights or not (named net_img_size.h5)")
    
    
    # Parse argument
    args = parser.parse_args()
    net_type = args.net
    epochs = args.epochs
    batch_size = args.batch_size
    gpu_number = args.gpu
    img_size = args.img_size
    
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    # Argument check
    if not (net_type in {"unet", "deeplab", "custom", "mrcnn"}):
        raise ValueError("netType should be either unet, deeplab, mrcnn and custom.")
        
    if net_type in {"unet", "deeplab", "custom"}:
        
        # Get config
        Config = cfg.Config()
           
        # COCO instance
        print("Reading COCO ground truth...")
        cocoGt = COCO(Config.COCO_training_ann_path)
        cocoValGt = COCO(Config.COCO_validation_ann_path)
        print("Finished")
    
    
        # Get all classes
        classes = len(cocoGt.getCatIds())

        id_to_index = dict()
        # There is a wired class of 0 in the feature map of type zero
        index_to_id = dict()

        # Because the id of COCO dataset starts from 92, we should project those id to index so that keras
        # utils can convert the segmentation map into one hot categorical encoding.
        for index, id in enumerate(cocoGt.getCatIds()):
            id_to_index[id] = index
            index_to_id[index] = id
    
        if net_type == "unet":
            model = basic_model.unet(input_size=(img_size, img_size, 3), classes=len(id_to_index))
        elif net_type == "deeplab":
            deeplab_model = basic_model.Deeplabv3(input_shape=(img_size, img_size, 3), classes = len(id_to_index), backbone="xception")
            output = KL.Activation("softmax")(deeplab_model.output)
            model = KM.Model(deeplab_model.input, output)
        elif net_type == "custom":
            model = model.custom_model(input_shape=(img_size, img_size, 3), classes=len(id_to_index))
            
            
    elif net_type == "mrcnn":
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
        
    elif net_type == "custom_inst":
        model = model.custom_inst_model(input_shape = input_size, classes = len(id_to_index))
        


        
    if net_type in {"unet", "deeplab", "custom"}:
    
        file_list = glob(Config.COCO_training_path + '*')
        val_list = glob(Config.COCO_validation_path + '*')
        
        if args.load_weights:
            try:
                model.load_weights(net_type + "_" + str(img_size) + ".h5")
                print("weights loaded!")
            except:
                print("weights not found!")

        checkpointer = KC.ModelCheckpoint(filepath= net_type + "_" + str(img_size) + ".h5", 
                                       verbose=1,
                                       save_best_only=True)

        #model.compile(optimizer = KO.Adam(clipvalue=2.), loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer = KO.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit_generator(data.generator(batch_size, file_list, (img_size, img_size), cocoGt, id_to_index, True),
                           validation_data=data.generator(batch_size, val_list, (img_size, img_size), cocoValGt, id_to_index, False),
                           validation_steps=10,
                           steps_per_epoch=100,
                           epochs=epochs,
                           use_multiprocessing=True,
                           workers=8,
                           callbacks=[checkpointer])
        print("Prediction start...")

        vfunc = np.vectorize(lambda index : index_to_id[index])

        anns = []

        # Convert into COCO annotation
        for i in trange(len(val_list)):
            image = val_list[i]
            image_id = int(image.replace(".jpg", '')[-12:])

            cropping_image, padding_dims, original_size = utils.padding_and_cropping(image, (img_size, img_size))
            cropping_image = preprocess_input(cropping_image, mode = "torch")

            result = model.predict(cropping_image)
            result = np.argmax(result, axis = 3)

            seg_result = utils.reverse_padding_and_cropping(result, padding_dims, original_size)
            seg_result = vfunc(seg_result)
            COCO_ann = cocostuffhelper.segmentationToCocoResult(seg_result, imgId = image_id)
            for ann in COCO_ann:
                ann["segmentation"]["counts"] = ann["segmentation"]["counts"].decode("ascii")# json can't dump byte string
            anns += COCO_ann

        with open("result.json", "w") as file:
            json.dump(anns, file)

        # Read result file
        # Test for fake result
        #resFile = fake_result

        # Evaluate result
        resFile = "result.json"
        cocoDt = cocoValGt.loadRes(resFile)
        cocoEval = COCOStuffeval(cocoValGt, cocoDt)
        cocoEval.evaluate()
        cocoEval.summarize()
        
    elif net_type == "mrcnn":
        # Training - Stage 1
        print("Training network heads")
        imgaug = data.CocoDataset().augmentation
        model.train(dataset_train, dataset_val,
                    learning_rate=Config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=imgaug)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=Config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=imgaug)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=Config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                        augmentation=imgaug)
        
        class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            
        config = InferenceConfig()
        config.display()

        model = mrcnn_model.MaskRCNN(mode="inference", config=config,
                                      model_dir=DEFAULT_LOGS_DIR)

        model_path = COCO_MODEL_PATH
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)
        # Validation dataset
        dataset_val = data.CocoDataset()
        val_type = "val"
        coco = dataset_val.load_coco("/mnt/dataset/experiment/COCO/", val_type, year="2017", return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format("100"))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int("100")) 

    
if __name__ == '__main__':
    main()