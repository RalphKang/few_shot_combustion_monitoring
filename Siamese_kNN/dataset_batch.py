import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

filenameToPILImage = lambda x: Image.open(x)
torch.manual_seed(111)


# %%
class dataset_siam(Dataset):
    def __init__(self, image_size, h_new, dataset_path, batch_size=5):
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_siam, self).__init__()

        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]
        self.h_new = h_new
        self.data_length = 0
        self.class_table = []
        self.class_length = 0
        self.train_cls_datas = {}
        self.batch_size = 5

        self.load_dataset()
        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((int(self.image_height), self.image_width)),
            transforms.CenterCrop(min(self.image_height, self.image_width)),
            transforms.Resize(int(h_new * random.uniform(1.1, 1.5))),  # train part zoom image
            transforms.RandomCrop(h_new),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.data_length // (self.batch_size * 2)

    def load_dataset(self):
        # load dataset from dataset_path
        train_path = self.dataset_path
        data_length = 0
        for character in os.listdir(train_path):
            # 遍历种类。
            self.class_table.append(character)  # store the class name
            train_lines = []
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                train_lines.append(os.path.join(character_path, image))
            data_length += len(train_lines)
            self.train_cls_datas[character] = train_lines  # dictionary, store the class name and the data inside it
        self.data_length = data_length
        self.class_length = len(self.class_table)

    def __getitem__(self, index):

        left, right, label = [], [], []

        for i in range(self.batch_size):
            class_type = np.random.randint(0, self.class_length, size=1)
            class_chose = self.class_table[class_type[0]]

            select1 = np.random.choice(self.train_cls_datas[class_chose], 1, replace=False)
            select2 = np.random.choice(self.train_cls_datas[class_chose], 1, replace=False)
            select3 = np.random.choice(self.train_cls_datas[class_chose], 1, replace=False)
            left.append(select1[0])
            right.append(select2[0])
            label.append([1])
            remained_class_table = self.class_table.copy()
            remained_class_table.pop(class_type[0])
            diff_class_type = np.random.randint(0, len(remained_class_table), size=1)
            diff_class_chose = remained_class_table[diff_class_type[0]]
            diff_select = np.random.choice(self.train_cls_datas[diff_class_chose], 1, replace=False)
            left.append(select3[0])
            right.append(diff_select[0])
            label.append([0])

        left_data = torch.stack([self.transform(os.path.join(name)) for name in left], 0)
        right_data = torch.stack([self.transform(os.path.join(name)) for name in right], 0)
        label = torch.tensor(label, dtype=torch.float32)
        return left_data, right_data, label


class dataset_siam_support(Dataset):
    def __init__(self, image_size, h_new, train_path, batch_size):
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_siam_support, self).__init__()

        self.dataset_path = train_path
        self.batch_size = batch_size
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
        # load dataset from dataset_path
        train_path = self.dataset_path
        data_length = 0
        label_type = 0
        for character in os.listdir(train_path):
            character_path = os.path.join(train_path, character)  # dir
            for image in os.listdir(character_path):
                self.image_list.append(os.path.join(character_path, image))  # store the read dir of images
                self.class_list.append(label_type)  # store the label of images
            label_type += 1  # update the label
            data_length += len(self.image_list)

    def __getitem__(self, index):
        label = torch.tensor(self.class_list[index])
        image = self.transform(os.path.join(self.image_list[index]))

        return image, label, self.image_list[index]  # return image, label, and train image dir


class dataset_siam_test_Online(Dataset):
    def __init__(self, image_size, h_new, test_path, batch_size):
        # !!! the batch_size here must be same with the batch size in support set
        # input image size is the original image size, h_new is the target image size, data_path is where image stored in
        super(dataset_siam_test_Online, self).__init__()

        self.dataset_path = test_path
        self.batch_size = batch_size
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
            transforms.Resize(int(h_new * 1.3)),  # simulate different size
            transforms.CenterCrop(h_new),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def load_dataset(self):
        # go through the dataset and store the image dir and label
        train_path = self.dataset_path
        data_length = 0
        label_type = 0
        for character in os.listdir(train_path):
            character_path = os.path.join(train_path, character)  # dir
            for image in os.listdir(character_path):
                self.image_list.append(os.path.join(character_path, image))  # store the read dir of images
                self.class_list.append(label_type)  # store the label of images
            label_type += 1  # update the label
            data_length += len(self.image_list)
        idx_list = np.arange(len(self.image_list))
        np.random.shuffle(idx_list)
        self.shuffle_idx_list = list(idx_list)

    def __getitem__(self, index):
        idx_picked = self.shuffle_idx_list[index]
        label = torch.tensor(self.class_list[idx_picked])
        image = self.transform(os.path.join(self.image_list[idx_picked]))
        image_batch = image.repeat(self.batch_size, 1, 1, 1)

        return image_batch, label, self.image_list[idx_picked]  # return image, label, and test image dir

