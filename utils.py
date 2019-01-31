from pycocotools import mask
import cv2 
import numpy as np

def padding_and_cropping(image, image_shape):
    """padding and cropping images

    Padding and cropping images accroding to the indicating image shape for inference.

    Args:
        - image(str): the file name of the image
        - image_shpae(tuple): the shape of required image

    Returns:
        - cropping_image(numpy array): return the resulting image
            * cropping_image [number of images, width of images, height of images, channels]
        - padding_dims(numpy array): stores the dimension of padding shape
            * padding_dims [width number of cropped images, height number of cropped images]
        - [im.shape[0], im.shape[1]]: original image shape
    """
    
    im = cv2.imread(image)
    # Mode 4 stands for any image between bgr and gray scale
    # Grayscale Gaurd
    #if im.ndim == 2:
    #    raise RuntimeError("Gray Scale Error")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    padding_shape = [(im.shape[0] // image_shape[0] + 1) * image_shape[0], (im.shape[1] // image_shape[1] + 1) * image_shape[1]]
    padding_dims = [im.shape[0] // image_shape[0] + 1, im.shape[1] // image_shape[1] + 1]
    padding_im = np.zeros((padding_shape[0], padding_shape[1], 3), dtype=np.uint8)
    padding_im[:im.shape[0], :im.shape[1]] = im

    cropping_image = np.zeros((padding_dims[0] * padding_dims[1], image_shape[0], image_shape[1], 3), dtype=np.uint8)

    for i in range(padding_dims[0]):
        for j in range(padding_dims[1]):
            cropping_image[i * padding_dims[1] + j] = padding_im[i * image_shape[0] : (i + 1) * image_shape[0], j * image_shape[1] : (j + 1) * image_shape[1]]
                
    return cropping_image, padding_dims, [im.shape[0], im.shape[1]]

def reverse_padding_and_cropping(cropping_image, padding_dims, original_size):
    
    padding_im = np.zeros((padding_dims[0] * cropping_image.shape[1], padding_dims[1] * cropping_image.shape[2]), dtype = np.uint8)
    for i in range(padding_dims[0]):
        for j in range(padding_dims[1]):
            padding_im[i * cropping_image.shape[1] : (i + 1) * cropping_image.shape[1], j * cropping_image.shape[2] : (j + 1) * cropping_image.shape[2]] = cropping_image[i * padding_dims[1] + j]
    origin_img = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)
    origin_img = padding_im[:original_size[0], :original_size[1]]
    
    return origin_img