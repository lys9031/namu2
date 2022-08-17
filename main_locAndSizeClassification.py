from classifyingSimilarities.locationAndSizeBasedClassification import LocationBasedClassification
from instanceSegmentation.segmentation import SegmentationModel
from classifyingSimilarities.utils import AnnotationInfo
import argparse
import os
pwd = (os.path.dirname(os.path.realpath(__file__)))

def getArgsParser():
    parser = argparse.ArgumentParser("location and size Classification script", add_help=False)
    parser.add_argument('--imgPath', type=str, help='target image path')
    parser.add_argument('--gpuNumber', type=str, help='set gpu number')
    parser.add_argument('--sortType', type=str, help='sort type')
    parser.add_argument('--saveData', type=str, help='saveData')
    return parser
def saveLocationClassificationDataframe():
    l = LocationBasedClassification()
    l.applyTypeOfChildrenToLC(pwd)
def instanceSegmentation(args):
    segClass = SegmentationModel()
    segClass.segmentationLaunch(args.gpuNumber, args, pwd)
    pred_classes = segClass.pred_classes
    return pred_classes, segClass
def getJsonAnnotationInfo(pred_classes):
    a = AnnotationInfo()
    nameClasses = a.getInfo(pred_classes)
    # print(a.getInfo(pred_classes))
    return nameClasses
def printPredictClassNameList(pred_classes):
    for index in range(len(pred_classes)):
        print(pred_classes[index])
def showInstanceSegResult(segClass):
    segClass.showImg()
def TargetData(nameClasses, segClass):
    targetData = segClass.getTargetData(nameClasses)
    return targetData, segClass.targetRatio
def calculateDistance(args, targetData, targetRatio):
    l = LocationBasedClassification()
    l.calcurateTargetImage(targetData, args.imgPath, args.sortType, targetRatio, pwd)

def main(args):
    if args.saveData =='save':
        saveLocationClassificationDataframe()
    pred_classes, segClass = instanceSegmentation(args)
    nameClasses = getJsonAnnotationInfo(pred_classes)
    printPredictClassNameList(pred_classes)
    showInstanceSegResult(segClass)
    targetData, segClass.targetRatio, = TargetData(nameClasses, segClass)
    calculateDistance(args, targetData, segClass.targetRatio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image Location and Size Based Similar Type scrips', parents=[getArgsParser()]
    )
    args = parser.parse_args()
    args.gpuNumber = '0'
    # args.imgPath = '/home/kmj21/ForestAndTree_v5.1/data/segmentation_data/5/5-1-2/11423.jpg'
    # args.sortType ='areaSize' #
    # args.sortType = 'distance'  # 'distance'
    main(args)



