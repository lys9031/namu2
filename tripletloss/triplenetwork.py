from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import random
import PIL.ImageOps
import csv
import os
import torch
from torch import optim
import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tripletloss.triplenetwork import *
from tripletloss.triplenetworkdataset import *
from tripletloss.resnet import *
from tripletloss.utils import *
from matplotlib import pylab as plt
from matplotlib import font_manager, rc
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib

class TripletNetwork():
    def __init__(self, pwd):
        self.pwd = pwd

    def tripletData(self, args):
        trainFolder = args.trainFolderforTriplet
        trainFolder = 'data/train/'
        trainFolderPath = os.path.join(self.pwd, trainFolder)
        trainFolderPath = dset.ImageFolder(root=trainFolderPath)

        testFolder = args.testFolderforTriplet
        testFolder = 'data/test/'
        testFolderPath = os.path.join(self.pwd, testFolder)
        testFolderPath = dset.ImageFolder(root=testFolderPath)

        totalFolder = args.totalFolderforTriplet
        totalFolder = 'data/train/'
        totalFolderPath = os.path.join(self.pwd, totalFolder)
        totalFolderPath = dset.ImageFolder(root=totalFolderPath)
        args.bathsizeforTriplet = 10

        train_triplet_dataset = TripletNetworkDataset(imageFolderDataset=trainFolderPath, transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]), should_invert=False)
        test_triplet_dataset = TripletNetworkDataset(imageFolderDataset=testFolderPath, train=False,
                                                     transform=transforms.Compose(
                                                         [transforms.Resize((224, 224)), transforms.ToTensor()]),
                                                     should_invert=False)
        total_triplet_dataset = TripletNetworkDataset(imageFolderDataset=totalFolderPath, train=False,
                                                      transform=transforms.Compose(
                                                          [transforms.Resize((224, 224)), transforms.ToTensor()]),
                                                      should_invert=False)

        train_dataloader = DataLoader(train_triplet_dataset,
                                      shuffle=True,
                                      num_workers=0,
                                      batch_size=args.bathsizeforTriplet)

        test_dataloader = DataLoader(test_triplet_dataset,
                                     shuffle=False,
                                     num_workers=0,
                                     batch_size=args.bathsizeforTriplet)

        total_dataloader = DataLoader(total_triplet_dataset,
                                      shuffle=False,
                                      num_workers=0,
                                      batch_size=args.bathsizeforTriplet)
        return train_dataloader, test_dataloader, total_dataloader

    def loadModel(self, pwd):
        model = resnet152().cpu()
        model.load_state_dict(torch.load(pwd+'/tripletloss/model/triplet_resnet152.pth'))
        return model

    def tripletTrain(self, args, train_dataloader, total_dataloader):

        for step, (anchor_img, anchor_label) in enumerate(total_dataloader):
            print(anchor_img.shape, anchor_label.shape)
            break
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 학습
        epochs = 50
        model = resnet152().to(device)

        criterion = nn.TripletMarginLoss(margin=5.0, p=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        counter = []
        loss_history = []
        iteration_number = 0

        from matplotlib import pylab as plt
        import numpy as np
        model.train()

        # train
        for epoch in range(epochs):
            print('epoch : ', epoch)
            running_loss = []
            for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_dataloader):
                optimizer.zero_grad()
                anchor_out = model(anchor_img.to(device))
                positive_out = model(positive_img.to(device))
                negative_out = model(negative_img.to(device))
                #     anchor_label = anchor_label.to(device)

                loss = criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss.item())

                print(loss)

                running_loss.append(loss.item())
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

        show_plot(counter, loss_history)
        torch.save(model.state_dict(), './tripletloss/model/triplet_resnet152.pth')  ## resnet152

    def tripletTrainVisualization(self, args, train_dataloader, model):
        # 학습 결과 시각화
        train_results = []
        labels = []

        model.cpu().eval()
        with torch.no_grad():
            for img, _, _, label in train_dataloader:
                train_results.append(model(img.cpu()))
                labels.append(label.cpu())

        train_results = np.concatenate(train_results)
        labels = np.concatenate(labels)
        # train_results.shape

        # visualization
        plt.figure(figsize=(10, 8), facecolor='azure')
        for label in np.unique(labels):
            tmp = train_results[labels == label]
            plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

        plt.legend()
        plt.show()

        np.save('./tripletloss/train_results/triplet_resnet152_2', train_results)
        np.save('./tripletloss/train_results/label_2', labels)
        train_results = np.load('./tripletloss/train_results/triplet_resnet152_2.npy')
        labels = np.load('./tripletloss/train_results/label_2.npy')

        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        for label in np.unique(labels):
            tmp = train_results[labels == label]
            ax.scatter(tmp[:, 0], tmp[:, 1], tmp[:, 2], marker='o', label=label)
            plt.suptitle('Resnet152 After learning')
            ax.view_init(5, 3)
            ax.set_xlabel('1st dim')
            ax.set_ylabel('2nd dim')
            ax.set_zlabel('3rd dim')
            ax.legend(loc='center right')

    def tripletSaveData(self, args, pwd, total_dataloader, model, ):
        img = Image.open(args.imgPath)
        img = img.resize((224, 224))

        tf = transforms.ToTensor()
        img_t = tf(img).unsqueeze(0)
        img.resize((224, 224))  # 새로운 이미지
        # 전체 데이터 셋의 특징 vector 구하기
        total_results = []

        model.eval()
        with torch.no_grad():
            for img, label in total_dataloader:
                total_results.append(model(img).numpy())

        total_results = np.concatenate(total_results)
        print(total_results.shape)

        np.save(pwd + '/tripletloss/train_results/total_results', total_results)

    def tripletLoad(self, args, pwd, total_dataloader, model):
        # 새로운 이미지(input) 가져오기
        img = Image.open(args.imgPath)
        img = img.resize((224, 224))
        tf = transforms.ToTensor()
        img_t = tf(img).unsqueeze(0)

        # 새 이미지의 특징 vector 구하기
        model.eval()
        with torch.no_grad():
            new_results = model(img_t).numpy()
        print(new_results.shape)

        total_results = np.load(pwd+'/tripletloss/train_results/total_results.npy')

        idx = []
        dist = []
        # class_name = []
        for i in range(0, len(total_results)):
            total = np.reshape(total_results[i], (1, 3))
            idx.append(i)
            dist.append(np.round(cosine_similarity(new_results, total).item(), 4))
            # class_name.append(re.findall('\w\w군',total_dataloader.dataset.images[i][0])[0])

        df = pd.DataFrame({'location': idx, 'distance': dist})
        top_n = df.sort_values(by='distance', ascending=False)[:6]

        path = []
        for i in top_n.index + 1:
            path.append(total_dataloader.dataset.images[i][0])

        # 유사도 높은 상위 5개 그림 출력
        fig1 = plt.figure(figsize=(4, 3))
        img = Image.open(args.imgPath)
        ax = fig1.add_subplot(111)
        ax.imshow(img)
        plt.axis('off')
        plt.title('Input Image', fontsize=12)
        # plt.show()

        fig = plt.figure(figsize=(5, 5))
        print('[Top 5 Similarity Images]')

        csv_resultPath_db = pwd + '/tripletloss/triplet_result/triplet_result_DB.csv'
        fileTripletResultDB = open(csv_resultPath_db, 'w', encoding='utf-8')

        index = 0
        for i in range(6):
            pathh = path[i]
            input_temp = pwd + args.imgPath

            if input_temp == pathh:
                continue
            else:
                if index >= 5:
                    continue
                else:
                    index += 1
                    img = Image.open(pathh)
                    ax = fig.add_subplot(5, 5, i + 1)
                    ax.imshow(img)
                    plt.title('Top {}'.format(i + 1), fontsize=25)
                    plt.axis('off')
                    i += 1

                    wr = csv.writer(fileTripletResultDB)
                    row = []
                    row.append(pathh)
                    wr.writerow(row)

        # plt.show()
        # plt.savefig('./top5_similarity.png')
