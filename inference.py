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
    parser = argparse.ArgumentParser(description = "Evaluate the model")
    parser.add_argument("--net", help="The type of net work which is either mrcnn, unet, deeplab or custom.",
                       required=True, default="unet")
    parser.add_argument("--img_size", required=True, type=int, help="The size of input image")
    parser.add_argument("--gpu_id", required=False, default="0", type=str, help="The id of the gpu used when training.")
    parser.add_argument("--weight_path", required-False, defalut=None, type=str, help="The name of trained weights. If not specified, the model will use 'net_imgSize.h5'. ")
    
    
    
    # Parse argument
    args = parser.parse_args()
    net_type = args.net
    gpu_number = args.gpu_id
    img_size = args.img_size
    weight_path = args.weight_path
    
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
        
    if net_type in {"unet", "deeplab", "custom"}:
    
        file_list = glob(Config.COCO_training_path + '*')
        val_list = glob(Config.COCO_validation_path + '*')
        
        if weight_path in None:
            model.load_weights(net_type + "_" + str(img_size) + ".h5")
            print("weights loaded!")
        else:
            model.load_weights(weight_path)
            print("weights loaded!")

        #model.compile(optimizer = KO.Adam(clipvalue=2.), loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer = KO.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
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

        # Evaluate result
        resFile = "result.json"
        cocoDt = cocoValGt.loadRes(resFile)
        cocoEval = COCOStuffeval(cocoValGt, cocoDt)
        cocoEval.evaluate()
        cocoEval.summarize()
        
if __name__ == '__main__':
    main()