from classifyingSimilarities.histogramBasedClassification import Histogram
import os
import argparse

typeOfChildren = ['5-1-1', '5-1-2', '6-1-1', '6-1-2', '7-1-1', '7-1-2']
typeOfChildren2 = ['class_0','class_1', 'class_2']

pwd = (os.path.dirname(os.path.realpath(__file__)))


def getArgsParser():
    parser = argparse.ArgumentParser("location and size Classification script", add_help=False)
    parser.add_argument('--imgPath', type=str, help='target Image path')
    parser.add_argument('--savePath', type=str, help='save histogram data')
    parser.add_argument('--top_k', type=str, help='top k info')
    parser.add_argument('--childernType', type=str, help='childernType')
    parser.add_argument('--saveData', type=str, help='saveData')
    return parser

def saveHistogramData(h):
    index = 0
    for type in typeOfChildren:
        args.dataSubPathforHistogram = type
        imgFolderPath = os.path.join(pwd +'/data/segmentation_data/' + typeOfChildren[index][0], type) +'/'
        h.setImgFolderPath(imgFolderPath)
        # h.draw_color_histogram_from_image()
        histogram_DB = h.build_histogram_db(args, type, pwd)
        index +=1
    for type2 in typeOfChildren2:
        args.dataSubPathforHistogram = type2
        imgFolderPath = pwd + '/data/train/' + type2 + '/'
        # print(imgFolderPath)
        h.setImgFolderPath(imgFolderPath)
        # h.draw_color_histogram_from_image()
        histogram_DB = h.build_histogram_db(args, type, pwd)
        index += 1

def buildTargetHistogram(args,h, pwd):
    h.build_targetimage_histogram(args, pwd)

def calculateDistance(h):
    result = h.search(args, args.childernType, pwd)
    # random_Sampling_histogram()
    return result
def showResult(h, result):
    h.show_result(args, pwd, result)
def main(args):
    h = Histogram()
    print('args.saveData:', args.saveData)
    if args.saveData =='save':
        saveHistogramData(h)
    buildTargetHistogram(args, h, pwd)
    result = calculateDistance(h)
    showResult(h, result)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Histogram Based Similar Type scrips', parents=[getArgsParser()]
    )
    args = parser.parse_args()
    args.savePath ='./classifyingSimilarities/data_histogram/'
    # args.imgPath= '/data/segmentation_data/5/5-1-1/13013.jpg'

    args.top_k = 6
    args.childernType ='5-1-1'
    main(args)
