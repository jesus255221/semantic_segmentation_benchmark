from pycocotools import mask
import cv2
import matplotlib.pylab as plt
import json
import numpy as np
from skimage import transform
from pycocotools.coco import COCO
from glob import glob
from os import listdir
from imgaug import augmenters as iaa
import random
from pycocotools.cocostuffhelper import cocoSegmentationToSegmentationMap
from keras.applications.resnet50 import preprocess_input
import keras
import utils
import config 

def generator(batch_size, image_list, image_shape, coco_instance, id_to_index, is_training):
    """Generator

    Generate the images for models to train

    Args:
        - batch_size: batch_size
        - image_list: the list of file name of images
        - image_shape: the target image shape
        - coco_instance: the ground truth of COCO dataset
        - id_to_index: dictionary project id to index
        - is_training: open or close data augmentation

    Returns:
        - all_img: shape: (batch_size, image_shape[0], image_shape[1], 3)
        - label: (batch_size, image_shape[0], image_shape[1], classes)
    """
    
    aug = iaa.SomeOf((0, 3), [
        iaa.Flipud(0.5), 
        iaa.Fliplr(0.5), 
        iaa.AdditiveGaussianNoise(scale=0.005*255),
        iaa.Grayscale(alpha=(0.0, 0.5)),
        iaa.GaussianBlur(sigma=(1.0)),
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)])
    
    def f(id):
        if id != 0:
            return id_to_index[id]
        else:
            return id_to_index[183]
    
    # Lambda function to convert id to index
    vfunc = np.vectorize(f)
    

    
    while True:

        # Random Shuffling
        random.shuffle(image_list)
        
        index = 0
        while index + batch_size < len(image_list):
            all_img = np.zeros((batch_size, image_shape[0], image_shape[1], 3), dtype=np.float32)
            label = np.zeros((batch_size, image_shape[0], image_shape[1], len(id_to_index)))
            i = 0
            while i < batch_size:
                image = image_list[index] 
                im = cv2.imread(image)
                # Mode 4 stands for any image between bgr and gray scale
                # Convert back to RGB
                # If the width or height is smaller than indicated size
                # or if the image is grayscale
                #if (im.shape[0] < image_shape[0] or im.shape[1] < image_shape[1]) or im.ndim == 2:
                #    i -= 1
                #    continue
                if (im.shape[0] < image_shape[0] or im.shape[1] < image_shape[1]):
                    index += 1
                    continue
                    
                im = cv2.cvtColor(im ,cv2.COLOR_BGR2RGB) 
                
                # Get the id containing 12 numbers (000000XXXXXX)
                lbl_id = int(image.replace(".jpg", '')[-12:])
                lbl = cocoSegmentationToSegmentationMap(coco_instance, lbl_id)
                lbl = vfunc(lbl.astype(np.uint8))
                
                # Resize
                #im = transform.rescale(im, 0.5)

                # Random Crop
                rnd_x = random.randint(0, im.shape[0] - image_shape[0])
                rnd_y = random.randint(0, im.shape[1] - image_shape[1])
                
                crop_im = im[rnd_x : rnd_x + image_shape[0], rnd_y : rnd_y + image_shape[1], :]
                crop_lbl = lbl[rnd_x : rnd_x + image_shape[0], rnd_y : rnd_y + image_shape[1]]
                
                # Convert to one hot
                crop_lbl = keras.utils.to_categorical(crop_lbl, num_classes = len(id_to_index))
                
                # Save data
                all_img[i] = crop_im
                label[i] = crop_lbl
                
                index += 1
                i += 1
            #if is_training:
            #    all_img = aug.augment_images(all_img)
            
            all_img = preprocess_input(all_img, mode = "torch")
            yield all_img, label
    
            
if __name__ == "__main__":
    
    cfg = config.Config()
    cocoGt = COCO(cfg.COCO_validation_ann_path)
    id_to_index = dict()
    id_to_index[0] = 0
    # id to index
    for index, id in enumerate(cocoGt.getCatIds(), 1):
        id_to_index[id] = index
    file_list = glob(cfg.COCO_validation_path + '*')
    
    for a, b in generator(16, file_list, (256, 256), cocoGt, id_to_index, False):
        print(a.shape)
        print(b.shape)