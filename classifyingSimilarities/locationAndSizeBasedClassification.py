from pycocotools.coco import COCO
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import csv
class LocationBasedClassification:

    def __init__(self):
        self.anchorX = 1000
        self.anchorY = 1000
        self.typeOfChildren = 'class_6_1_1'
        self.classInfo = {'1': 'tree', '2': 'person', '3': 'house'}
        self.annFile = './data/segmentation_data/5/5-1-1/exports/output_5_1_1.json'
        self.imageFolderPath='./data/segmentation_data/5/5-1-1'
        self.categories_nms = [4, 5, 6]
        self.bboxData_x  = []
        self.bboxData_y  = []
        self.pathData  = []
        self.catData   = []
        self.typeOfChildren = ['class_5_1_1', 'class_5_1_2', 'class_6_1_1', 'class_6_1_2', 'class_7_1_1', 'class_7_1_2']
        self.areaSize=[]
        self.imgRatio = []
        self.numberOfPerson = []
        self.numberOfTree = []
        self.numberOfHouse = []


    def setTypeOfChildren(self, type):
        self.typeOfChildren = type

    def setInfo(self):

        if self.typeOfChildren == 'class_5_1_1':
            self.classInfo = {'1': 'tree', '2': 'person', '3': 'house'}
            self.annFile = '/data/segmentation_data/5/5-1-1/exports/output_5_1_1.json'
            self.imageFolderPath = '/data/segmentation_data/5/5-1-1'
            print('class_5_1_1')
        elif self.typeOfChildren == 'class_5_1_2':
            self.classInfo = {'1': 'tree', '2': 'person', '3': 'house'}
            self.annFile = '/data/segmentation_data/5/5-1-2/exports/output_5_1_2.json'
            self.imageFolderPath = '/data/segmentation_data/5/5-1-2'
            print('class_5_1_2')
        elif self.typeOfChildren == 'class_6_1_1':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            self.annFile = '/data/segmentation_data/6/6-1-1/exports/output_6_1_1.json'
            self.imageFolderPath = '/data/segmentation_data/6/6-1-1'
            print('class_6_1_1')
        elif self.typeOfChildren == 'class_6_1_2':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            self.annFile =  '/data/segmentation_data/6/6-1-2/exports/output_6_1_2.json'
            self.imageFolderPath =  '/data/segmentation_data/6/6-1-2'
            print('class_6_1_2')
        elif self.typeOfChildren == 'class_7_1_1':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            self.annFile =  '/data/segmentation_data/7/7-1-1/exports/output_7_1_1.json'
            self.imageFolderPath =  '/data/segmentation_data/7/7-1-1'
            print('class_7_1_1')
        elif self.typeOfChildren == 'class_7_1_2':
            self.classInfo = {'4': 'person', '5': 'tree', '6': 'house'}
            self.annFile =  '/data/segmentation_data/7/7-1-2/exports/output_7_1_2.json'
            self.imageFolderPath =  '/data/segmentation_data/7/7-1-2'
            print('class_7_1_2')

    def getCOCO(self, pwd):
        coco = COCO(pwd + self.annFile)
        cats = coco.loadCats(coco.getCatIds())
        return coco, cats

    def showCategories_nms(self, pwd):
        coco, cats = self.getCOCO(pwd)
        # categories
        categories_nms = [cat['name'] for cat in cats]
        print('COCO categories: {}'.format(' '.join(categories_nms)))
        # supercategory
        supercategory_nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: {}'.format(' '.join(supercategory_nms)))
    def setCategoriesNms(self, pwd):
        coco, cats = self.getCOCO(pwd)
        if self.typeOfChildren  == 'class_7_1_2' or self.typeOfChildren  == 'class_7_1_2':
            self.categories_nms = [4, 5, 6]
        else:
            self.categories_nms = [cat['name'] for cat in cats]
    def getImagesIDS(self, pwd):
        # get all images containing given categories, select one at random
        coco, cats = self.getCOCO(pwd)
        catIds = coco.getCatIds(catNms=self.categories_nms)

        if self.typeOfChildren == 'class_5_1_1':
            imgIds = coco.getImgIds(catIds=2)
        elif self.typeOfChildren == 'class_5_1_2':
            imgIds = coco.getImgIds(catIds=2)
        else:
            imgIds = coco.getImgIds(catIds=catIds)
        return coco, cats, catIds, imgIds
    def saveDataFrame(self, pwd):
        print('save')
        # print(self.bboxData_x)
        # print(len(self.bboxData_x))
        sum = {}
        sum.update({"imgRatio"      : self.imgRatio})
        sum.update({"imgPath"       : self.pathData})
        sum.update({"categories"        : self.catData})

        sum.update({"bboxData_x"       : self.bboxData_x})
        sum.update({"bboxData_y"       : self.bboxData_y})

        sum.update({"areaSize": self.areaSize})



        sum.update({"numberOfPerson": self.numberOfPerson})
        sum.update({"numberOfTree": self.numberOfTree})
        sum.update({"numberOfHouse": self.numberOfHouse})

        df = pd.DataFrame(sum, columns=['imgRatio','bboxData_x', 'bboxData_y', 'imgPath', 'categories', 'distance', 'areaSize', 'areaSizeDiff', 'imgRatioDiff', 'numberOfPerson','numberOfTree','numberOfHouse'])
        print(df.head(10))
        #dfName = './classifyingSimilarities/data_location_size/' + str(self.typeOfChildren) + '.csv'

        dfName = pwd + '/classifyingSimilarities/data_location_size/db.csv'
        print('dfName:', dfName)


        df.to_csv(str(dfName), mode='w', encoding = 'utf-8')

    def calcurateTargetImage(self, targetData, imgPath, sortType, targetRatio, pwd) :


        numberOftarget_person = 0
        numberOftarget_house  = 0
        numberOftarget_tree   = 0

        for i in range(len(targetData)):
            if targetData[i][0] =='person':
                numberOftarget_person +=1
            elif targetData[i][0] =='tree':
                numberOftarget_tree+=1
            elif targetData[i][0] == 'house':
                numberOftarget_house+=1

        # print('numberOftarget_person: {}, numberOftarget_tree: {}, numberOftarget_house: {}'.format(numberOftarget_person, numberOftarget_tree, numberOftarget_house))

        # for type in self.typeOfChildren:
        ratioX = targetData[0][3][1]/self.anchorX
        ratioY =  targetData[0][3][0]/self.anchorY

        fileName = pwd + '/classifyingSimilarities/data_location_size/db.csv'

        df = pd.read_csv(fileName)
        sum_dic = []

        for indx in range(len(df['bboxData_x'])):
            for targetData_indx in range(len(targetData)):
                if df['categories'][indx] == targetData[targetData_indx][0]:
                    bboxData_x = df.loc[df.index[indx], 'bboxData_x']
                    bboxData_y = df.loc[df.index[indx], 'bboxData_y']

                    target_bboxData_x = targetData[targetData_indx][1] * ratioX
                    target_bboxData_y = targetData[targetData_indx][2] * ratioY

                    locationComparison = np.sqrt(np.sum((bboxData_x - target_bboxData_x)**2) + np.sum((bboxData_y - target_bboxData_y)**2))

                    addDist_dic = df.loc[df.index[indx]]
                    addDist_dic['distance'] = locationComparison

                    ############################################################
                    # areaSize calcuration
                    areaSize = df.loc[df.index[indx], 'areaSize']
                    # targetAreaSize = targetData[targetData_indx][4]
                    targetAreaSize = targetData[targetData_indx][4]
                    areaSizeDiff = abs(areaSize-targetAreaSize)
                    addDist_dic['areaSizeDiff'] = areaSizeDiff
                    addDist_dic['imgRatioDiff'] = abs(targetRatio - df.loc[df.index[indx], 'imgRatio'])

                    ############################################################
                    sum_dic.append(addDist_dic)


        df2 = pd.DataFrame(sum_dic, columns=['bboxData_x', 'bboxData_y', 'imgPath', 'categories', 'distance', 'areaSize' , 'areaSizeDiff', 'imgRatioDiff', 'numberOfPerson','numberOfTree','numberOfHouse'])
        print(df2.head(10))

        dfName = pwd +'/classifyingSimilarities/data_location_size/db_new.csv'


        df2.to_csv(str(dfName), mode='w', encoding='utf-8')

        # filter_person= df2.categories =='person'
        # filter_tree  = df2.categories =='tree'
        # filter_house = df2.categories =='house'

        # df2 = df2.loc[filter_person, :]
        # df2= df2.loc[filter_tree,:]
        # df2 = df2.loc[filter_house, :]

        if sortType=='distance':
           locationclassification_data = df2.sort_values(by=['distance'], axis=0, ascending=True)
           csv_resultPath_db = pwd +'/classifyingSimilarities/result_location_size/distance_result_DB.csv'
           fileResultDB = open(csv_resultPath_db, 'w', encoding='utf-8')
           path_temp= pwd+ '/classifyingSimilarities/result_location_size/distance_result.png'

        elif sortType=='areaSize':
           locationclassification_data = df2.sort_values(by=['areaSize'], axis = 0, ascending=True)
           csv_resultPath_db = pwd+ '/classifyingSimilarities/result_location_size/size_result_DB.csv'
           fileResultDB = open(csv_resultPath_db, 'w', encoding='utf-8')
           path_temp = pwd + '/classifyingSimilarities/result_location_size/size_result.png'

        print('before locationclassification_data : ', locationclassification_data.head(10))
        print('numberOftarget_person {}, numberOftarget_tree {}, numberOftarget_house {}'.format(numberOftarget_person,numberOftarget_tree,numberOftarget_house))
        locationclassification_data =locationclassification_data.loc[locationclassification_data['imgRatioDiff']<0.7]
        locationclassification_data =locationclassification_data.loc[locationclassification_data['numberOfPerson']  == numberOftarget_person]
        locationclassification_data = locationclassification_data.loc[locationclassification_data['numberOfTree']   == numberOftarget_tree]
        locationclassification_data = locationclassification_data.loc[locationclassification_data['numberOfHouse']  == numberOftarget_house ]

        print('after locationclassification_data : ', locationclassification_data.head(10))
        if len(locationclassification_data) > 10:
            plt.figure(figsize=(10,10))
            for i in range(6):
                image_index = i + 1
                if i==0:  # input 값
                    # plt.title('input')
                    plt.title('input image', fontsize=12)

                    image = img.imread(imgPath) # orignal mode
                    # image = img.imread(pwd+imgPath) # integration mode
                    plt.imshow(image)


                else:
                    # title = "Top {} Image".format(image_index)
                    # plt.title('output')
                    # image = img.imread(locationclassification_data.iloc[i].imgPath)


                    wr = csv.writer(fileResultDB)
                    row = []
                    print('locationclassification_data.iloc[i].imgPath:', locationclassification_data.iloc[i].imgPath)
                    row.append(locationclassification_data.iloc[i].imgPath)
                    wr.writerow(row)


                    if i==1:
                        plt.title('input image', fontsize=12)
                        plt.axis('off')
                    else:
                        plt.axis('off')
                        plt.title('Tops {}'.format(i-1), fontsize=12)

                # plt.subplot(1, 6, image_index)
                # plt.axis('off')
                # plt.title('Tops {}'.format(i ), fontsize=12)
                # plt.imshow(image)
            path = path_temp
            # plt.savefig(path)
            # plt.show()

    def applyTypeOfChildrenToLC(self, pwd):
        for type in self.typeOfChildren:
            self.setTypeOfChildren(type)
            self.locationBasedClassificationLaunch(pwd)

    def generateDataframe(self, pwd):
        coco, cats, catIds, imgIds = self.getImagesIDS(pwd)
        for imgIndex in imgIds:
            # imgIds = coco.getImgIds(imgIds = imgIndex)
            img = coco.loadImgs(imgIndex)[0]
            imgSPath = img['path']


            if self.typeOfChildren == 'class_5_1_1':
                imgPath = self.imageFolderPath + imgSPath[15:]
                print('5-1-1 imgPath:', imgPath)
            elif self.typeOfChildren == 'class_5_1_2':
                imgPath = self.imageFolderPath + imgSPath[15:]
                print('5-1-2 imgPath:', imgPath)
            elif self.typeOfChildren == 'class_6_1_1':
                imgPath = self.imageFolderPath + imgSPath[16:]
                print('6-1-1 imgPath:', imgPath)
            elif self.typeOfChildren == 'class_6_1_2':
                imgPath = self.imageFolderPath + imgSPath[16:]
                print('6-1-2 imgPath:', imgPath)
            elif self.typeOfChildren == 'class_7_1_1':
                imgPath = self.imageFolderPath + imgSPath[16:].replace(' ', '')
                print('7-1-1 imgPath:', imgPath)
            elif self.typeOfChildren == 'class_7_1_2':
                imgPath = self.imageFolderPath + imgSPath[15:].replace(' ', '')
                print('7-1-2 imgPath:', imgPath)



            imgPath = pwd + imgPath
            # print('pwd:', pwd)
            # print('imgPath:', imgPath)

            # imshow
            I = io.imread(imgPath)

            plt.axis('off')
            # plt.imshow(I)

            plt.axis('off')
            #
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            coco.showAnns(anns)

            # path = '/home/kaeri/ForestAndTree_v5.1/annotation_img/result_{}.png'.format(imgIndex)
            # plt.savefig(path)
            #
            # plt.show()


            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            # imgWidth
            # imgHeight
            numberOfPerson = 0
            numberOfTree   = 0
            numberOfHouse   = 0

            for annsIndex in range(len(anns)):
                categoryNumber = anns[annsIndex]['category_id']
                categoryId = self.classInfo[str(categoryNumber)]

                if categoryId=='person':
                    numberOfPerson+=1
                elif categoryId=='tree':
                    numberOfTree += 1
                elif categoryId == 'house':
                    numberOfHouse += 1


                ratioX = img['width']  / self.anchorX
                ratioY = img['height'] / self.anchorY


                # object number ...

                bboxData_x_temp = (anns[annsIndex]['bbox'][0] + anns[annsIndex]['bbox'][2] / 2) * ratioX
                bboxData_y_temp = (anns[annsIndex]['bbox'][1] + anns[annsIndex]['bbox'][3] / 2) * ratioY
                self.imgRatio.append(img['width']/img['height'])
                self.bboxData_x.append(bboxData_x_temp)
                self.bboxData_y.append(bboxData_y_temp)
                self.catData.append(categoryId)
                self.pathData.append(imgPath)
                # self.areaSize.append(anns[annsIndex]['area'])
                self.areaSize.append( abs(anns[annsIndex]['bbox'][2] - anns[annsIndex]['bbox'][0]) * abs(anns[annsIndex]['bbox'][3] -anns[annsIndex]['bbox'][1]))
                # print('size: ',  abs(anns[annsIndex]['bbox'][2] - anns[annsIndex]['bbox'][0]) * abs(anns[annsIndex]['bbox'][3] -anns[annsIndex]['bbox'][1]))

            # print('person: {}, tree:{}, haouse:{}'.format(numberOfPerson,numberOfTree, numberOfHouse ))
            for annsIndex in range(len(anns)):
                self.numberOfPerson.append(numberOfPerson)
                self.numberOfTree.append(numberOfTree)
                self.numberOfHouse.append(numberOfHouse)
    def locationBasedClassificationLaunch(self, pwd):

        self.setInfo()
        self.getCOCO(pwd)
        self.showCategories_nms(pwd)
        self.setCategoriesNms(pwd)
        self.generateDataframe(pwd)
        self.saveDataFrame(pwd)

if __name__ == "__main__":
    l = LocationBasedClassification()
    l.locationBasedClassificationLaunch()