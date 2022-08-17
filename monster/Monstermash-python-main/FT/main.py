import argparse
from segToMonsterSetting import segToMonster
import os
def getArgsParser():
    parser = argparse.ArgumentParser("location and size Classification script", add_help=False)
    # parser.add_argument('--imgPath', type=str, help='target image path')
    return parser

def main(args, jpg_img):
    s = segToMonster()
    for imgInx in range(len(jpg_img)):
        print('imgInx:', imgInx)
        s.makeinstanceSegToOrg(args, jpg_img, imgInx)
        s.makeInstanceSegToSeg(args, jpg_img, imgInx)

    s.makeResultZipFile()



    s.deactivateAnaconda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image Location and Size Based Similar Type scrips', parents=[getArgsParser()]
    )
    args = parser.parse_args()
    file_list = os.listdir('./data')

    jpg_img = []
    for file in file_list:
        if '.jpg'  in file:
            jpg_img.append(file)

    print('jpg_img:', jpg_img)

    # args.img = './data/mask_tree.jpg'
    main(args, jpg_img)