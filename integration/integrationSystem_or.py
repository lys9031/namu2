import os
import csv
import pandas as pandasForSortingCSV
import csv ,operator
from matplotlib import pyplot as plt
from PIL import Image
import shutil
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

pwd = (os.path.dirname(os.path.realpath(__file__)))

def initSetting():
    split_pwd = pwd.split('/')
    index =0
    path_sum = '/'
    for name in split_pwd:
        if index >0 and index <4:

            path_sum +=name +'/'
        index +=1
    print('name_sum:', path_sum)

    pathList = [path_sum +'integration/csvFile', path_sum + 'tripletloss/triplet_result' , path_sum + 'classifyingSimilarities/data_histogram', path_sum + 'classifyingSimilarities/data_location_size',
                path_sum + 'classifyingSimilarities/result_histogram',  path_sum + 'classifyingSimilarities/result_location_size']


    for path in pathList:
        print('path:', path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)


def eleminateDuplicateFileName(inputFileName, outFileName):
    lines = set()
    outfile= open(outFileName, "w")
    for line in open(inputFileName, "r"):
        if line not in lines:
            outfile.write(line)
            lines.add(line)
    outfile.close()

def sccoring(inputFileName, scoreList):
    scoringName_ = inputFileName.split('/')[2].split('.csv')[0]
    scoreList_ = list(scoreList.values())
    thr_distance = 0
    thr_size = 0
    thr_histogram = 0
    thr_triplet = 0

    with open(inputFileName, 'r') as csvinput:
        with open('./csvFile/integrate.csv', 'a') as csvoutput:
            writer = csv.writer(csvoutput)
            for row in csv.reader(csvinput):
                if scoringName_ =='distance_result_DB':
                    writer.writerow(row + [str(int(scoreList_[0]) - thr_distance)])
                    thr_distance += 10
                elif scoringName_ =='size_result_DB':
                    writer.writerow(row + [str(int(scoreList_[1]) - thr_size)])
                    thr_size += 10
                elif scoringName_ == 'histogram_result_DB':
                    writer.writerow(row + [str(int(scoreList_[2]) - thr_histogram)])
                    thr_histogram += 10
                elif scoringName_ == 'triplet_result_DB':
                    writer.writerow(row + [str(int(scoreList_[3]) - thr_triplet)])
                    thr_triplet += 10
def sumScoring():
    duplicate_result = []
    result = []
    all_result= []
    with open('./csvFile/integrate.csv', 'r') as csvinput:
        for row in csv.reader(csvinput):
            # print(row[0])
            # print('index :', index)
            all_result.append(row)
            if row[0] not in result:
                result.append(row[0])
            else: # 중복
                duplicate_result.append(row[0])
    # print(duplicate_result)

    duplicate_result_new = []
    for line in duplicate_result:
        if line not in duplicate_result_new:
            duplicate_result_new.append(line)

    # print('duplicate_result_new:', duplicate_result_new)

    result_sum = []
    for line in all_result:
        if line[0] not in duplicate_result_new: # 중복 되지 않은 것
            result_sum.append(line)

    # 중복 더해주기
    sum = 0
    sum_index = 0
    result_dup = []
    for dup_line in duplicate_result_new:

        for all_line in all_result:
            if dup_line == all_line[0]:
                sum +=int(all_line[1])
                sum_index+=1
        # print(dup_line)
        # print(sum)
        result_dup.append([dup_line, sum])
        sum = 0
    # print(result_dup)
    subtitle =[['filename', 'score']]
    all_list = subtitle+ result_dup + result_sum

    # print(result_sum)
    # print(all_list)

    with open("./csvFile/final_integrate.csv",'w') as file:
        write = csv.writer(file)
        write.writerows(all_list)

def sorting():
    csvData = pandasForSortingCSV.read_csv("./csvFile/final_integrate.csv")

    csvData.sort_values("score",
                        axis=0,
                        ascending=False,
                        inplace=True)
    print("\nAfter sorting:")
    print(csvData)
    # csvData.to_csv
    csvData.to_csv("./csvFile/sorting_final_integrate.csv", encoding='utf-8', index=False)


def showImg(path):
    titles = ['input', 'Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5']

    index= 0

    plt.figure(figsize=(20, 3))
    img = Image.open(path)
    plt.subplot(1, 6, index+1)
    plt.title(titles[index])
    plt.imshow(img)

    with open('./csvFile/sorting_final_integrate.csv', 'r') as csvinput:
        for row in csv.reader(csvinput):

            index = index + 1
            if index == 1:
                continue
            elif index == 7:
                break
            else:
                print(row[0])
                print(index)
                img = Image.open(row[0])
                plt.subplot(1, 6, index )
                plt.title(titles[index-1])
                plt.imshow(img)

    plt.show()


def main():

    FirstTimeSetting = True
    path = '/home/lys/ForestAndTree_v7.1/data/segmentation_data/5/5-1-2/23154.jpg'


    if FirstTimeSetting:
        print('FirstTimeSetting')
        initSetting()

        imgPath = '--imgPath ' + path
        saveData =' --saveData save'

        # Histogram
        commandHistogram = 'python ../main_HistogramBasedClassification.py ' + imgPath + saveData
        os.system(commandHistogram)
        print('Histogram 완료')

        # Triplet loss
        commandTriplet = 'python ../main_Triplet.py ' + imgPath + saveData
        os.system(commandTriplet)
        print('Triplet loss 완료')
        #
        # location
        sortType = ' --sortType distance'
        commandTriplet = 'python ../main_locAndSizeClassification.py ' + imgPath + sortType + saveData
        os.system(commandTriplet)
        print('location 완료 ')
        # areaSize
        sortType = ' --sortType areaSize'
        commandTriplet = 'python ../main_locAndSizeClassification.py ' + imgPath + sortType + saveData
        os.system(commandTriplet)
        print('size 완료 ')
    else:

        print('Setting')

        imgPath = '--imgPath ' + path
        saveData = ' --saveData no_save'

        # Histogram
        commandHistogram = 'python ../main_HistogramBasedClassification.py ' + imgPath + saveData
        os.system(commandHistogram)
        print('Histogram 완료')
        # Triplet loss
        commandTriplet = 'python ../main_Triplet.py ' + imgPath + saveData
        os.system(commandTriplet)
        print('Triplet loss 완료')
        #
        # location
        sortType = ' --sortType distance'
        commandTriplet = 'python ../main_locAndSizeClassification.py ' + imgPath + sortType + saveData
        os.system(commandTriplet)
        print('location 완료 ')
        # areaSize
        sortType = ' --sortType areaSize'
        commandTriplet = 'python ../main_locAndSizeClassification.py ' + imgPath + sortType + saveData
        os.system(commandTriplet)
        print('size 완료 ')

    input_csvList =["../classifyingSimilarities/result_location_size/distance_result_DB.csv","../classifyingSimilarities/result_location_size/size_result_DB.csv", "../classifyingSimilarities/result_histogram/histogram_result_DB.csv", "../tripletloss/triplet_result/triplet_result_DB.csv"]
    csvList =["./csvFile/distance_result_DB.csv", "./csvFile/size_result_DB.csv", "./csvFile/histogram_result_DB.csv", "./csvFile/triplet_result_DB.csv"]

    scoreList = {'distance': 100, 'size' : 100 , 'histogram' : 100, 'triplet' : 100}

    for i in range(len(csvList)):
        eleminateDuplicateFileName(input_csvList[i], csvList[i])

    for i in range(len(csvList)):
        sccoring(csvList[i], scoreList)

    sumScoring()
    sorting()
    # showImg(path)

if __name__=="__main__":
    main()

