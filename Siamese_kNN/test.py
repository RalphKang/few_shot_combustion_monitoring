import heapq
import time
import cv2
import numpy as np
from pandas import DataFrame
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from component_module import *
from dataset_batch import dataset_siam_support, dataset_siam_test_Online
from nets.siamese_new import Siamese
import argparse
import os

"""
    This file is used to test the model on the test set.
    """


def main_test(image_type: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./IR_dataset/train_set', help='train dataset dir')
    parser.add_argument('--test_dir', type=str, default='./IR_dataset/test_set', help='test dataset dir')
    parser.add_argument('--datasave_dir', type=str, default='./data_save', help='where to store the processed data')
    parser.add_argument('--batch_size', type=int, default=120, help='the size of training set')
    parser.add_argument('--image_size', type=list, default=[215, 161, 3], help='original image size')
    parser.add_argument('--target_size', type=list, default=[84, 84, 3], help='target image size')
    parser.add_argument('--model_path', type=str, default='./model_dir/ir_model.pth', help='model path')
    parser.add_argument('--online_show', type=bool, default=True, help='whether to show the online result')
    parser.add_argument('--store_result', type=bool, default=True, help='whether to store the result')
    args = parser.parse_args()
    assert image_type in ['IR', 'visible'], 'image type must be IR or visible'
    if image_type == 'IR':
        args.train_dir = './IR_dataset/train_set'
        args.test_dir = './IR_dataset/test_set'
        args.model_path = './model_dir/ir_model.pth'
        args.image_size = [215, 161, 3]
    else:
        args.train_dir = './visible_dataset/train_set'
        args.test_dir = './visible_dataset/test_set'
        args.model_path = './model_dir/visible_model.pth'
        args.image_size = [480, 640, 3]

    # %% data prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %% data part
    train_dataset = dataset_siam_support(args.image_size, args.target_size[0], args.train_dir,
                                         batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = dataset_siam_test_Online(args.image_size, args.target_size[0], args.test_dir,
                                            batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)  # batch size is one here, don't change

    # %% model part
    net = Siamese(input_shape=args.target_size, pretrained=False)
    net.load_state_dict(torch.load(args.model_path))
    net.to(device)
    # %% test part
    all_test_embedding = []
    test_label_list = []
    pred_label_list = []
    wrong_pred = []
    start = time.process_time()
    for test_index, test_data in enumerate(test_loader):
        image_test, test_label, test_image_dir = test_data
        # print('label',label_test)
        right = image_test.squeeze()
        pred, train_embedding, test_embedding = test_for_label(net, train_loader, right, device)
        # ----------------------------------------
        if args.online_show:
            ori_img = cv2.imread(test_image_dir[0])
            img = cv2.resize(ori_img, (1080, 720))
            font = cv2.FONT_HERSHEY_DUPLEX
            text = 'label: class-' + str(int(test_label)) + '   prediction: class-' + str(int(pred))
            cv2.putText(img, text, (44, 44), font, 1, (255, 255, 255), 2, lineType=5)
            out_win = "prediction"
            cv2.imshow(out_win, img)
            # cv2.imshow('prediction',img)
            cv2.waitKey(200)
            test_label.numpy()
        # -------------------------------------------------
        if test_label != pred:
            wrong_pred.append(test_image_dir[0])
        test_label_list.append(test_label)
        pred_label_list.append(pred.numpy())
        all_test_embedding.append(test_embedding[0].cpu().numpy())
        # break
    end = time.process_time()
    print("the total time for {} images= {} ms, average time per image = {} ms".format(len(test_label_list),
                                                                                       (end - start) * 1000,
                                                                                       (end - start) * 1000 / len(
                                                                                           test_label_list)))

    # %% store the result
    if args.store_result:
        all_test_embedding = np.array(all_test_embedding)
        train_embedding = np.array(train_embedding.cpu())
        test_label_list = np.array(test_label_list)
        pred_label_list = np.array(pred_label_list)

        np.savetxt(os.path.join(args.datasave_dir, 'test_fea_2.txt'), all_test_embedding)
        np.savetxt(os.path.join(args.datasave_dir, 'support_fea.txt'), train_embedding)
        np.savetxt(os.path.join(args.datasave_dir, 'label_list.txt'), test_label_list)
        np.savetxt(os.path.join(args.datasave_dir, 'pred_list.txt'), pred_label_list)

        wrong_pred_df = DataFrame(wrong_pred)
        wrong_pred_df.to_csv('wrong_pred.csv')


if __name__ == '__main__':
    main_test(image_type='IR')
