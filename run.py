# Import configuration
import config

# Import Unet and DeepLab
import basic_model
# Import custom model
import model
# Import utilities
import utils

from data import generator
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
from glob import glob
from tqdm import trange



def main(input_size = (192, 192, 3)):
    """main function

    Main function... (what do you expect me to say...)

    Args:
        - input_size: the size of image

    Returns:
        - none
    """
    # Get config
    Config = config.Config()
    
    
    # Main function for evaluate
    parser = argparse.ArgumentParser(description = "A testing framework for detection, semantic seg and instance seg.")
    parser.add_argument("--ann", help="The type of the annotation which is either seg or bbox.", 
                       required=True, default="seg")
    parser.add_argument("--net", help="The type of net work which is either unet, deeplab or custom.",
                       required=True, default="unet")
    parser.add_argument("--epochs", required=False, default=500, type=int)
    parser.add_argument("--batch_size", required=False, default=16, type=int)
    parser.add_argument("--gpu", required=False, default="0", type=str, help="The id of the gpu used when training.")
    

    
    # Parse argument
    args = parser.parse_args()
    ann_type = args.ann
    net_type = args.net
    epochs = args.epochs
    batch_size = args.batch_size
    gpu_number = args.gpu
    
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    # Argument check
    if not (ann_type in {"seg", "bbox"}):
        raise ValueError("annType should be either seg or bbox")
    if not (net_type in {"unet", "deeplab", "custom"}):
        raise ValueError("netType should be either unet, deeplab or custom.")
    if (net_type == "unet" or net_type == "deeplab") and not ann_type == "seg":
        raise ValueError("unet and deeplab are segmentation models.")
        
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
        model = basic_model.unet(input_size = input_size, classes = len(id_to_index))
    elif net_type == "deeplab":
        model = basic_model.Deeplabv3(input_shape = input_size, classes = len(id_to_index), backbone = "xception")
    elif net_type == "custom":
        model = model.custom_model(input_shape = input_size, classes = len(id_to_index))
    
    file_list = glob(Config.COCO_training_path + '*')
    val_list = glob(Config.COCO_validation_path + '*')
    
    #model.load_weights(net_type + "_256.h5")
    
    checkpointer = KC.ModelCheckpoint(filepath= net_type + "_256.h5", 
                                   verbose=1,
                                   save_best_only=True)
    
    model.compile(optimizer = KO.Adam(clipvalue=0.5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit_generator(generator(batch_size, file_list, (input_size[0], input_size[1]), cocoGt, id_to_index, True),
                       validation_data=generator(batch_size, val_list, (input_size[0], input_size[1]), cocoValGt, id_to_index, False),
                       validation_steps=10,
                       steps_per_epoch=100,
                       epochs=epochs,
                       use_multiprocessing=True,
                       workers=8,
                       callbacks=[checkpointer])
    print("Prediction start...")
    
    vfunc = np.vectorize(lambda index : index_to_id[index])
    
    anns = []
    
    # Transfer into COCO annotation
    for i in trange(len(val_list)):
        image = val_list[i]
        image_id = int(image.replace(".jpg", '')[-12:])
        
        cropping_image, padding_dims, original_size = utils.padding_and_cropping(image, (input_size[0], input_size[1]))
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

    
if __name__ == '__main__':
    main();