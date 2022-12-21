
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from random import shuffle
import random

filenameToPILImage = lambda x: Image.open(x)
torch.manual_seed(111)

#%%
class dataset_pt(Dataset):
    def __init__(self, image_size, h_new, dataset_path, way=6, shot=5, query=5):
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_pt, self).__init__()

        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        self.h_new = h_new
        self.way = way
        self.shot = shot
        self.query = query
        self.data_length = 0
        self.class_table = []
        self.train_cls_datas = {}

        self.load_dataset()
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((int(self.image_height),self.image_width)),
            transforms.CenterCrop(min(self.image_height, self.image_width)),
            transforms.Resize(int(h_new * random.uniform(1.1,1.5))),
            transforms.RandomCrop(h_new),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.data_length // (self.way * self.shot)

    def load_dataset(self):

        train_path = self.dataset_path
        data_length = 0
        for character in os.listdir(train_path):
            # 遍历种类。
            self.class_table.append(character)
            train_lines = []
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                train_lines.append(os.path.join(character_path, image))
            data_length += len(train_lines)
            self.train_cls_datas[character] = train_lines
        self.data_length = data_length

    def __getitem__(self, index):

        support, query = [], []
        for clss in self.class_table:
            select = np.random.choice(self.train_cls_datas[clss], self.shot + self.query, replace=False)
            support.extend(select[:self.shot])
            query.extend(select[self.shot:])
        support = torch.stack([self.transform(os.path.join(name))/255 for name in support], 0)
        query = torch.stack([self.transform(os.path.join(name))/255 for name in query], 0)
        return support, query

class dataset_pt_proto(Dataset):
    def __init__(self, image_size, h_new, dataset_path, way=6, shot=5):
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_pt_proto, self).__init__()

        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        self.h_new = h_new
        self.way = way
        self.shot = shot
        self.data_length = 0
        self.class_table = []
        self.train_cls_datas = {}

        self.load_dataset()
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.CenterCrop(min(self.image_height, self.image_width)),
            transforms.Resize(int(h_new * 1.3)),
            transforms.CenterCrop(h_new),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.data_length // (self.way * self.shot)

    def load_dataset(self):
    # load dataset
        train_path = self.dataset_path
        data_length = 0
        for character in os.listdir(train_path):
            # 遍历种类。
            self.class_table.append(character)
            train_lines = []
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                train_lines.append(os.path.join(character_path, image))
            data_length += len(train_lines)
            self.train_cls_datas[character] = train_lines
        self.data_length = data_length

    def __getitem__(self, index):

        support, query = [], []
        for clss in self.class_table:
            select = self.train_cls_datas[clss]
            support.extend(select)
            query.extend(select[self.shot:])
        support = torch.stack([self.transform(os.path.join(name))/255 for name in support], 0)
        return support


class dataset_pt_test(Dataset):
    def __init__(self, image_size, h_new, dataset_path, way=6, query=5):
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_pt_test, self).__init__()

        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        self.h_new = h_new
        self.way = way
        self.query = query
        self.data_length = 0
        self.class_table = []
        self.train_cls_datas = {}

        self.load_dataset()
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((int(self.image_height), self.image_width)),
            transforms.CenterCrop(min(self.image_height, self.image_width)),
            transforms.Resize(int(h_new * 1.3)),
            transforms.CenterCrop(h_new),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.data_length // (self.way * self.query)

    def load_dataset(self):
    # load dataset
        train_path = self.dataset_path
        data_length = 0
        for character in os.listdir(train_path):
            # 遍历种类。
            self.class_table.append(character)
            train_lines = []
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                train_lines.append(os.path.join(character_path, image))
            data_length += len(train_lines)
            self.train_cls_datas[character] = train_lines
        self.data_length = data_length

    def __getitem__(self, index):

        support, query = [], []
        for clss in self.class_table:
            select = np.random.choice(self.train_cls_datas[clss], self.query, replace=False)
            query.extend(select)
        query = torch.stack([self.transform(os.path.join(name))/255 for name in query], 0)
        return query

class dataset_pt_test_Online(Dataset):
    def __init__(self, image_size, h_new, dataset_path):
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_pt_test_Online, self).__init__()

        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        self.h_new = h_new
        self.data_length = 0
        self.class_list = []
        self.image_list = []
        self.shuffle_idx_list = []


        self.load_dataset()
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((int(self.image_height), self.image_width)),
            transforms.CenterCrop(min(self.image_height, self.image_width)),
            transforms.Resize(int(h_new * 1.3)),
            transforms.CenterCrop(h_new),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def load_dataset(self):
    # load dataset
        train_path = self.dataset_path
        data_length = 0
        label_type = 0
        for character in os.listdir(train_path):
            character_path = os.path.join(train_path, character) # dir
            for image in os.listdir(character_path):
                self.image_list.append(os.path.join(character_path, image)) # store the read dir of images
                self.class_list.append(label_type) # store the label of images
            label_type +=1 # update the label
            data_length += len(self.image_list)
        idx_list = np.arange(len(self.image_list))
        np.random.shuffle(idx_list)
        self.shuffle_idx_list = list(idx_list)

    def __getitem__(self, index):
        idx_picked = self.shuffle_idx_list[index]
        label=self.class_list[idx_picked]
        image=self.transform(os.path.join(self.image_list[idx_picked])) / 255
        return image, label, self.image_list[idx_picked]



