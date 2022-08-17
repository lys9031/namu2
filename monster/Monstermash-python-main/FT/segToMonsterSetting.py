import cv2
import numpy as np
import os

class segToMonster:
    def __init__(self):
        pass
    def makeinstanceSegToOrg(self, args, jpg_img, imgInx ):
        path_dir = os.path.dirname(os.path.realpath(__file__))
        imgPath = path_dir + '/data/{}'.format(str(jpg_img[imgInx]))
        print(imgPath)
        src = cv2.imread(imgPath)


        if src.shape[2]==3:
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        width, height = gray.shape
        img_white = np.ones((width, height), dtype=np.uint8) * 255
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i in range(len(contours)):
            cv2.drawContours(img_white, [contours[i]], 0, (0, 0, 0), 2)

            path_imwrite = './data/result/_org_00{}.png'.format(str(imgInx))
            cv2.imwrite(path_imwrite, img_white)
            # cv2.waitKey(0)

    def makeInstanceSegToSeg(self, args, jpg_img, imgInx):
        path_dir = os.path.dirname(os.path.realpath(__file__))
        imgPath = path_dir + '/data/{}'.format(str(jpg_img[imgInx]))

        src = cv2.imread(imgPath, cv2.IMREAD_COLOR)

        if (len(src.shape) < 3):
            print('gray')
        elif len(src.shape) == 3:
            print('Color(RGB)')
        else:
            print('others')

        height, width = src.shape[:2]
        for y in range(0, height):
            for x in range(0, width):
                b = src.item(y, x, 0)
                g = src.item(y, x, 1)
                r = src.item(y, x, 2)
                if b < 100 :
                    src.itemset(y, x, 0, 0)
                if b > 200:
                    src.itemset(y, x, 0, 255)
                if g < 100:
                    src.itemset(y, x, 1, 0)
                if g > 200:
                    src.itemset(y, x, 1, 255)
                if r < 100:
                    src.itemset(y, x, 2, 0)
                if r > 200:
                    src.itemset(y, x, 2, 255)
        # cv2.imshow("src", src)
        cv2.imwrite('./data/result/_seg_00{}.png'.format(imgInx), src)
        # cv2.waitKey(0)
    def makeResultZipFile(self ):
        path = os.path.dirname(os.path.realpath(__file__)) +'/data/result'
        os.chdir(path)
        cmd = 'zip -r result.zip . *'
        os.system(cmd)


    def deactivateAnaconda(self):
        path = os.path.dirname(os.path.abspath(__file__))
        print(path)
        path = 'bash '+path+'/monster_bash.sh'
        os.system(path)



