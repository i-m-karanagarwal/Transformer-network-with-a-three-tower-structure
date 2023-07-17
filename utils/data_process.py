import os
from random import shuffle

from torch.utils.data import Dataset
import torch
import numpy as np
from utils.load_data import PathSpectogramFolder, patients, loadSpectogramData,nSeizure

OutputPathModels = "./EggModels"
class MyDataset(Dataset):

    def __init__(self,
                 dataset: str):
        """
        Dataset object of training dataset and testing dataset
        :param path: dataset path
        :param dataset: Distinguish whether to obtain a training set or a test set
        """
        super(MyDataset, self).__init__()
        self.dataset = dataset  # Choose to get the test set or the training set
        self.train_len, \
        self.test_len, \
        self.input_len, \
        self.channel_len, \
        self.output_len, \
        self.train_dataset, \
        self.test_dataset, \
        self.train_label, \
        self.test_label, \
        self.hz = self.pre_option()

    def __getitem__(self, index):
        if self.dataset == 'train':
            return self.train_dataset[index], self.train_label[index]
        elif self.dataset == 'test':
            return self.test_dataset[index], self.test_label[index]

    def __len__(self):
        if self.dataset == 'train':
            return self.train_len
        elif self.dataset == 'test':
            return self.test_len

    # data preprocessing
    def pre_option(self):
        print(patients)
        for indexPat in range(0, len(patients)):
            print('Patient ' + patients[indexPat])
            if not os.path.exists(OutputPathModels + "resultPat" + patients[indexPat] + "/"):
                os.makedirs(OutputPathModels + "resultPat" + patients[indexPat] + "/")
            interictalSpectograms,preictalSpectograms,nSeizure = loadSpectogramData(indexPat)
            print('Spectograms data loaded')
            filesPath = []
            for i in range(0, nSeizure):
                filesPath.extend(interictalSpectograms[i])
                filesPath.extend(preictalSpectograms[i])
            shuffle(filesPath)
            print(filesPath)
            train_label = []
            test_label = []
            start = 0
            end = 90
            from_ = int(len(filesPath) / 100 * start)
            to_ = int(len(filesPath) / 100 * end)
            for i in range(from_, int(to_)):
                f = filesPath[i]
                # print(f)
                x = np.load(PathSpectogramFolder + f)
                if (i == from_):
                    train = x
                else:
                    train = np.concatenate((train, x), axis=0)
                if ('P' in f):
                    for j in range(x.shape[0]):
                        train_label.append(1)
                else:
                    for j in range(x.shape[0]):
                        train_label.append(0)
            start = 90
            end = 100

            from_ = int(len(filesPath) / 100 * start)
            to_ = int(len(filesPath) / 100 * end)
            for i in range(from_, int(to_)):
                f = filesPath[i]
                x = np.load(PathSpectogramFolder +  f)
                if (i == from_):
                    test = x
                else:
                    test = np.concatenate((test, x), axis=0)

                if ('P' in f):
                    for j in range(x.shape[0]):
                        test_label.append(1)
                else:
                    for j in range(x.shape[0]):
                        test_label.append(0)
            train_len = train.shape[0]
            test_len = test.shape[0]
            output_len = len(tuple(set(train_label)))

            # Sample Time Channel Hertz

            train = torch.Tensor(train).permute(0, 2, 1, 3)
            test = torch.Tensor(test).permute(0, 2, 1, 3)
            train_label = torch.Tensor(train_label)
            test_label = torch.Tensor(test_label)

            step = train[0].shape[0]
            channel = train[0].shape[1]
            hz = train[0].shape[-1]

        return train_len, test_len, step, channel, output_len, train, test, train_label, test_label, hz

