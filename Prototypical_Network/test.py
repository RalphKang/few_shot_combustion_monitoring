import time

import cv2
import numpy as np
from pandas import DataFrame
from torch.utils.data import DataLoader
import torch

from dataset_module import dataset_pt_proto, dataset_pt_test_Online
from component_module import EuclideanLoss
from nets.encoder_module import VGG_embedding
import argparse


def main_test(image_type: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=6, help='how many classes in one episode')
    parser.add_argument('--shot', type=int, default=20, help='how many samples in one class for training set')
    parser.add_argument('--query', type=int, default=20, help='how many samples in one class for test set')
    parser.add_argument('--image_size', type=list, default=[640, 480, 3], help='the original image size')
    parser.add_argument('--target_size', type=list, default=[84, 84, 3], help='the target image size')
    parser.add_argument('--train_dir', type=str, default='./visual_dataset/train_set', help='the path of train set')
    parser.add_argument('--test_dir', type=str, default='./visual_dataset/test_set', help='the path of test set')
    parser.add_argument('--model_path', type=str, default='model_best.pth', help='the path of model')
    parser.add_argument('--record_dir', type=str, default='./data_save/', help='the path to save the record')
    parser.add_argument('--online_show', type=bool, default=True, help='whether to show the online test')
    parser.add_argument('--store_result', type=bool, default=False, help='whether to store the result')

    args = parser.parse_args()
    assert image_type in ['IR', 'visible'], 'image type must be IR or visible'
    if image_type == 'IR':
        args.train_dir = './IR_dataset/train_set'
        args.test_dir = './IR_dataset/test_set'
        args.model_path = './model_dir/ir_model.pth'  # well_trained model is in the dir of ./model_dir
        args.image_size = [215, 161, 3]
    else:
        args.train_dir = './visible_dataset/train_set'
        args.test_dir = './visible_dataset/test_set'
        args.model_path = './model_dir/visible_model.pth'  # well_trained model is in the dir of ./model_dir
        args.image_size = [480, 640, 3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # %% data part
    proto_dataset = dataset_pt_proto(args.image_size, args.target_size[0], dataset_path=args.train_dir, way=args.way,
                                     shot=args.shot)
    proto_loader = DataLoader(proto_dataset, batch_size=1)
    test_dataset = dataset_pt_test_Online(args.image_size, args.target_size[0], dataset_path=args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # %%
    net = VGG_embedding(input_shape=args.target_size, way=args.way, shot=args.shot, query=args.query, pretrained=False)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)

    criterion = EuclideanLoss(way=args.way, shot=args.shot, query=args.query)
    criterion.to(device)

    # %% get prototype
    with torch.no_grad():
        net.eval()
        for _, train_data in enumerate(proto_loader):
            supports = train_data.squeeze()
            supports = supports.to(device)
            supports_feat = net(supports)

    # %% test
    label_list = []
    pred_list = []
    wrong_pred = []
    right_mark = 0

    start = time.process_time()
    for epoch, data_online in enumerate(test_loader):
        test_image, test_label, image_dir_tuple = data_online
        image_dir = image_dir_tuple[0]
        test_image = test_image.to(device)
        queries_feat = net(test_image, mode='test_one')
        dist, nothing = criterion(supports_feat, queries_feat)
        prop = dist
        pred = prop.argmax(dim=1)
        pred = pred.to('cpu')
        if pred == test_label:
            right_mark += 1
        else:
            wrong_pred.append(image_dir)
        label_list.append(test_label)
        pred_list.append(pred)
        queries_feat = queries_feat.to('cpu')
        if epoch == 0:
            fead_test = queries_feat
        else:
            fead_test = torch.cat((fead_test, queries_feat))

        # online show----------------------------------------
        if args.online_show:
            ori_img = cv2.imread(image_dir)
            img = cv2.resize(ori_img, (720, 480))
            font = cv2.FONT_HERSHEY_DUPLEX
            text = 'label: class-' + str(int(test_label) + 1) + '   prediction: class-' + str(int(pred) + 1)
            cv2.putText(img, text, (44, 44), font, 1, (255, 255, 255), 2, lineType=5)
            if str(int(test_label)) != str(int(pred)):
                cv2.putText(img, text, (44, 44), font, 1, (0, 0, 255), 2, lineType=5)
            out_win = "prediction"
            cv2.imshow(out_win, img)
            # cv2.imshow('prediction',img)
            cv2.waitKey(200)  # stay 200ms, for human to see

    end = time.process_time()
    accuracy = right_mark / (epoch + 1)
    supports_feat = supports_feat.to('cpu')
    print(
        "the total time for {} images= {} ms, average time per image = {} ms".format(len(label_list),
                                                                                     (end - start) * 1000,
                                                                                     (end - start) * 1000 / len(
                                                                                         label_list)))
    # %% store the result
    if args.store_result:
        test_feat = np.array(fead_test)
        support_feat = np.array(supports_feat)
        pred_list = np.array(pred_list)
        label_list = np.array(label_list)

        np.savetxt(args.record_dir + 'support_fea.txt', support_feat)
        np.savetxt(args.record_dir + 'test_fea_2.txt', test_feat)
        np.savetxt(args.record_dir + 'pred_list.txt', pred_list)
        np.savetxt(args.record_dir + 'label_list.txt', label_list)
        wrong_pred_df = DataFrame(wrong_pred)
        wrong_pred_df.to_csv('wrong_pred.csv')


if __name__ == '__main__':
    main_test(image_type='visible')
