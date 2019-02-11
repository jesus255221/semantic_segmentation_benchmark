# Semantic segmentation testing framework
## Goal
Construct a automatically training and testing framework to evaluate the performance of the model.
This framework includes the following modules.
1. preprocessing
    1. random cropping
    2. image augmentation using [imgaug](https://github.com/aleju/imgaug)
2. model construction
    1. unet
    2. deeplab with weights pretrained on pascal VOC
    3. mrcnn
3. evaluation
    1. official [COCOapi](https://github.com/cocodataset/cocoapi)

## Dependencies
1. `opencv2 4.0.0`
2. `imgaug 0.2.6`
3. `keras 2.2.4`
4. `python3`
5. `tensorflow 1.11.0`

## Installation
No need to install

## Run example
`python run.py [-h] --net NET [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--gpu GPU] [--img_size IMG_SIZE]`

`-h, --help            show this help message and exit`
  `--net NET             The type of net work which is either unet, deeplab or custom.`
  `--epochs EPOCHS`
  `--batch_size BATCH_SIZE`
  `--gpu GPU             The id of the gpu used when training.`
  `--img_size IMG_SIZE   The size of input image`

### Models
* `basic_model.py`

consists of [unet](https://github.com/zhixuhao/unet) and deeplab for `run.py` to train and evaluate.

* `model.py`

consists of custom model.
When defining your own model, you should implement class `custom_model` in the `model.py` which accept `input_size` which is the input image size and `classes` which is the number of classes to be classified. Class custom_model should return a keras model instance so that `run.py` could build model from it.

* `mrcnn/`

consists of all the files of mrcnn


### Configuration files
* `config.py`

Use a class to manage the configuration

### Datasets
* `data.py`

contains a generator with imgaug for custom model, unet and deeplab and a dataset reader for mrcnn.

### Utilities
* `utils.py`

contains some utilties that will be used by any file. It contains padding_and_cropping function for inference stage.

## Result
Result will be showed on the screen


