import os
from tripletloss.triplenetwork import *
from tripletloss.triplenetwork import TripletNetwork
pwd = (os.path.dirname(os.path.realpath(__file__)))\

def getArgsParser():
    parser = argparse.ArgumentParser("Triplet", add_help=False)
    parser.add_argument('--trainFolderforTriplet', type=str, help='trainFolderforTriplet')
    parser.add_argument('--testFolderforTriplet', type=str, help='trainFolderforTriplet')
    parser.add_argument('--totalFolderforTriplet', type=str, help='totalFolderforTriplet')
    parser.add_argument('--bathsizeforTriplet', type=int, help='bathsizeforTriplet')
    parser.add_argument('--imgPath', type=str, help='input Image path')
    parser.add_argument('--saveData', type=str, help='saveData')
    return parser

def tripletFunction(args):
    triplet = TripletNetwork(pwd)
    train_dataloader, test_dataloader, total_dataloader = triplet.tripletData(args)
    # triplet.tripletTrain(args, train_dataloader, total_dataloader)




    model = triplet.loadModel(pwd)
    # triplet.tripletTrainVisualization(args, train_dataloader, model)
    if args.saveData =='save':
        triplet.tripletSaveData(args, pwd, total_dataloader, model)
    triplet.tripletLoad(args, pwd, total_dataloader, model)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='FT 유사유형 프로그램 입니다.', parents=[getArgsParser()])
    args = parser.parse_args()

    # args.imgPath = './data/train/class_0/11324.jpg'


    tripletFunction(args)




