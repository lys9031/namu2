from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# get image
path = '/home/kmj21/db/FT/test_namu/datasets/1214_total200/28372.jpg'
im = cv2.imread(path)

# Create config
cfg = get_cfg()
cfg.merge_from_file("./config/mask_rcnn_R_101_FPN_3x_namu.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "./namu_model/model_final.pth"
# Create predictor
predictor = DefaultPredictor(cfg)
# Make prediction
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
########################################################################
mask = outputs["instances"].pred_masks.to("cpu").numpy()
mask = mask.astype(np.uint8)

####################################################
numberOfClass = mask.shape[0]

for i in range(numberOfClass): # 3
    for j in range(mask.shape[1]):  # 495
        for m in range(mask.shape[2]): # 717
            if i==0 and mask[i][j][m] > 0:
                mask[i][j][m] = 255
            elif i==1 and mask[i][j][m] > 0:
                mask[i][j][m] = 255
            elif i==2 and mask[i][j][m] > 0:
                mask[i][j][m] = 255

####################################################
mask_people = np.expand_dims(mask[0], axis=2)
mask_tree = np.expand_dims(mask[1], axis=2)
mask_house = np.expand_dims(mask[2], axis=2)
####################################################
path_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = path_dir + '/result_image/'

cv2.imwrite(path_dir + 'mask_people.jpg', mask_people)
cv2.imwrite(path_dir + 'mask_tree.jpg', mask_tree)
cv2.imwrite(path_dir + 'mask_house.jpg', mask_house)

path_dir2 = '../Monstermash-python-main/FT/data/'

cv2.imwrite(path_dir2 + 'mask_people.jpg', mask_people)
cv2.imwrite(path_dir2 + 'mask_tree.jpg', mask_tree)
cv2.imwrite(path_dir2 + 'mask_house.jpg', mask_house)
####################################################


