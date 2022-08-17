import os
import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import random

class Histogram:
    def __init__(self):
        self.imgFolderPath = ''
        self.imgList = []
    def setImgFolderPath(self, imgFolderPath):
        self.imgFolderPath = imgFolderPath

    def draw_color_histogram_from_image(self):
        imgList =[]
        for imgName in os.listdir(self.imgFolderPath):
            if 'jpg' or 'png' or 'jpeg':
                imgList.append(imgName)
        randomChoiceImg = random.choice(imgList)
        print(randomChoiceImg)

        imgPath = os.path.join(self.imgFolderPath,randomChoiceImg)
        print(imgPath)

        img = Image.open(imgPath)
        cv_image = cv2.imread(imgPath)


        f = plt.figure(figsize=(30, 30))


        im1 = f.add_subplot(1, 2, 1)
        im1.grid(False)
        im1.imshow(img)
        im1.set_title("Image")

        # # Histogram 시각화
        im2 = f.add_subplot(1,2,2)
        color = ('b','g','r')     # RGB->BGR
        for i,col in enumerate(color):
            # image에서 i번째 채널의 히스토그램을 뽑아서(0:blue, 1:green, 2:red)
            histr = cv2.calcHist([cv_image],[i],None,[256],[0,256])
            im2.plot(histr,color = col)

        im2.grid(False)
        im2.set_title("Histogram")
        plt.show()

    def get_histogram(self, image):
        histogram = []
        channels = image.shape[2]

        assert channels ==3, 'image channel should be 3 channel '
        # Create histograms per channels, in 4 bins each.
        for i in range(3):
            channel_histogram = cv2.calcHist(images=[image],
                                             channels=[i],
                                             mask=None,
                                             histSize=[256],  # 히스토그램 구간 256
                                             ranges=[0, 255])
            histogram.append(channel_histogram)
        histogram = np.concatenate(histogram)
        histogram = cv2.normalize(histogram, histogram)
        return histogram


    def build_histogram_db(self, args, type, pwd):
        format = [".jpg", ".png", ".jpeg"]
        file_list = os.listdir(self.imgFolderPath)
        # csv_file = args.savePath + str(type) +'.csv'
        csv_file =pwd +  args.savePath[1:] + 'histogramDB.csv'

        for file_name in tqdm(file_list ):
            if file_name.endswith(tuple(format)):
                file_path = os.path.join(self.imgFolderPath, file_name)
                image = cv2.imread(file_path)
                histogram = self.get_histogram(image)
                size = histogram.shape[0]


                with open(csv_file, 'a') as f:
                    wr = csv.writer(f)
                    row = []
                    for i in range(size+1):
                        if i==0:
                            row.append( self.imgFolderPath+file_name)
                        else:
                            row.append(histogram[i-1][0])
                    wr.writerow(row)

    def build_targetimage_histogram(self, args, pwd):
        targetPath = args.imgPath
        image = cv2.imread(targetPath)
        histogram = self.get_histogram(image)
        csv_file = pwd + args.savePath[1:] + 'targetData.csv'
        print(csv_file)

        size = histogram.shape[0]
        with open(csv_file, 'w') as f:
            wr = csv.writer(f)
            row = []
            for i in range(size + 1):
                if i == 0:
                    row.append(targetPath)
                else:
                    row.append(histogram[i - 1][0])
            wr.writerow(row)


    def target_histogram(self, targetPath, histogram_db):
        return histogram_db[targetPath]

    def search(self, args, type, pwd):
        csv_path_db = pwd + '/classifyingSimilarities/data_histogram/histogramDB.csv'
        csv_path_target = pwd + '/classifyingSimilarities/data_histogram/targetData.csv'
        results = {}

        fileHistDB = open(csv_path_db, 'r', encoding='utf-8')
        rdr = csv.reader(fileHistDB)

        fileTargetDB = open(csv_path_target, 'r', encoding='utf-8')
        target_rdr = csv.reader(fileTargetDB)

        target_list=[]
        for target in target_rdr:
            target_list=target

        for line_db in rdr:

            histogram_numpy = np.array((line_db[1:]), dtype=np.float32)
            histogram_numpy = np.expand_dims(histogram_numpy, axis=1)

            target_numpy = np.array((target_list[1:]), dtype=np.float32)
            target_numpy = np.expand_dims(target_numpy, axis=1)

            file_name = line_db[0]
            # try:
            distance = cv2.compareHist(H1=target_numpy, H2=histogram_numpy, method=cv2.HISTCMP_CHISQR)
            # except Exception as ex:
            #     error_msg = f"[calcurate hist] {str(ex)} "
            #     print(error_msg)

            results[file_name] = distance

        results = dict(sorted(results.items(), key=lambda item: item[0])[0:args.top_k])
        # results = dict(sorted(results.items(), key=lambda item: item[1])[0:args.top_k])

        fileHistDB.close()
        return results

    def show_result(self, args, pwd, result):
        csv_resultPath_db = pwd + '/classifyingSimilarities/result_histogram/histogram_result_DB.csv'
        fileHistResultDB = open(csv_resultPath_db, 'w', encoding='utf-8')

        f = plt.figure(figsize=(10, 3))

        im = f.add_subplot(1, len(result)+1, 1)
        im.grid(False)
        targetImg_path = args.imgPath
        img = Image.open(targetImg_path)
        im.imshow(img)

        plt.axis('off')
        plt.title('input image', fontsize=12)

        print('[Top 5 Similarity images]')
        for idx, filename in enumerate(result.keys()):
            im = f.add_subplot(1, len(result)+1, idx + 2)
            im.grid(False)
            img = Image.open(filename)
            im.imshow(img)
            plt.axis('off')
            plt.title('Tops {}'.format(idx + 1), fontsize=12)

            # write path csv top5 similarity images
            wr = csv.writer(fileHistResultDB)
            row = []
            row.append(filename)
            wr.writerow(row)


        plt.savefig(pwd + '/classifyingSimilarities/result_histogram/result.png')
        # plt.show()

