from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
import numpy as np

class SegmentationModel:
    def __init__(self):
        self.path = ''
        self.cfg = get_cfg()
        self.predictor = DefaultPredictor(self.cfg)
        self.predictionOutput ={}
        self.pred_classes=[]
        self.targetRatio =0

    def setGPU(self, numberOfGPU):
        os.environ["CUDA_VISIBLE_DEVICES"] = numberOfGPU

    def setTargetImgPath(self, imgPath, pwd):
        # self.path = pwd + imgPath  # integrate mode
        print('imgPath:', imgPath)
        self.path = imgPath  # original mode

    def setConfig(self):
        self.cfg = get_cfg()

        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_yaml = path_dir + '/config/mask_rcnn_R_101_FPN_3x_namu.yaml'
        path_weight = path_dir + '/namu_model/model_final.pth'

        self.cfg.merge_from_file(path_yaml)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = path_weight

    def getPrediction(self):
        predictor = DefaultPredictor(self.cfg)


        img = cv2.imread(self.path)  # original mode


        self.predictionOutput  = predictor(img)
        targetHeight = img.shape[0]
        targetWidth = img.shape[1]
        self.targetRatio = targetWidth/targetHeight

        return self.predictionOutput

    def showImg(self):
        predictor = DefaultPredictor(self.cfg)
        img = cv2.imread(self.path)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        path_dir = os.path.dirname(os.path.realpath(__file__))
        cv2.imwrite(path_dir + '/result_img/instanceSeg_result.jpg', out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)

    def getTargetData(self, nameClasses):
        targetData = []
        targetData.append([])
        for ix_box in range(len(self.predictionOutput["instances"].pred_boxes)):
            coordinates = self.predictionOutput["instances"].pred_boxes[ix_box].tensor.cpu().numpy()
            x1 = coordinates[0][0]
            x2 = coordinates[0][2]
            target_bbox_x = (x1 + x2) / 2
            y1 = coordinates[0][1]
            y2 = coordinates[0][3]
            target_bbox_y = (y1 + y2) / 2
            name = nameClasses[ix_box]

            targetData[ix_box].append(name)
            targetData[ix_box].append(target_bbox_x)
            targetData[ix_box].append(target_bbox_y)

            targetData[ix_box].append(self.predictionOutput["instances"].image_size)  # height, width

            targetData[ix_box].append(abs(x1 - x2) * abs(y1 - y2))  # height, width
            # targetData[ix_box].append(np.count_nonzero(self.predictionOutput['instances'].pred_masks[ix_box].cpu().numpy()))

            if ix_box != (len(self.predictionOutput["instances"].pred_boxes) - 1):
                targetData.append([])
        return targetData

    def segmentationLaunch(self, gpuNumber, args, pwd):
        # print('args imgpath', args.imgPath)
        self.setGPU(gpuNumber)
        self.setTargetImgPath(args.imgPath, pwd)
        self.setConfig()
        outputs = self.getPrediction()
        self.pred_classes = outputs['instances'].pred_classes.cpu().tolist()


if __name__ == "__main__":
    S = SegmentationModel()
    S.segmentationLaunch(gpuNumber='1', imgPath='../data/segmentation_data/6/6-1-1/22740.jpg' )
    pred_classes = S.pred_classes
