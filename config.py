############################################################
#  Unet and Deeplab Config
############################################################

class Config():
    # path of COCO dataset
    COCO_training_path = "/mnt/dataset/experiment/COCO/train2017/"
    COCO_validation_path = "/mnt/dataset/experiment/COCO/val2017/"
    COCO_training_ann_path = "/mnt/dataset/experiment/COCO/annotations/stuff_train2017.json"
    COCO_validation_ann_path = "/mnt/dataset/experiment/COCO/annotations/stuff_val2017.json"
    
    # path of fake result
    fake_result = "/mnt/dataset/experiment/COCO/annotations/instances_val2014_fakebbox100_results.json"

# ############################################################
# #  MrCNN Configurations
# ############################################################

from mrcnn.config import Config as mrcnn_config

class CocoConfig(mrcnn_config):
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
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    # Path to trained weights file
    COCO_MODEL_PATH = "/workspace/david/automatic_testing/mask_rcnn_coco.h5"

    # Directory to save logs and model checkpoints, if not provided
    # through the command line argument --logs
    DEFAULT_LOGS_DIR = "/workspace/david/automatic_testing/mrcnn_logs"
    DEFAULT_DATASET_YEAR = "2017"
    
    # path of COCO dataset
    COCO_training_ann_path = "/mnt/dataset/experiment/COCO"
    COCO_validation_ann_path = "/mnt/dataset/experiment/COCO"