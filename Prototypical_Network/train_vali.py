import argparse

import numpy as np
from torch.utils.data import DataLoader

from component_module import *
from dataset_module import dataset_pt
from nets.encoder_module import VGG_embedding


def main_train(image_type: str):
    # %% hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=6, help='how many classes in one episode')
    parser.add_argument('--shot', type=int, default=5, help='how many samples in one class for support set')
    parser.add_argument('--query', type=int, default=5, help='how many samples in one class for query set')
    parser.add_argument('--image_size', type=list, default=[640, 480, 3], help='the original image size')
    parser.add_argument('--target_size', type=list, default=[84, 84, 3], help='the target image size')
    parser.add_argument('--train_dir', type=str, default='./visual_dataset/train_set', help='the path of train set')
    parser.add_argument('--vali_dir', type=str, default='./visual_dataset/vali_set', help='the path of test set')
    parser.add_argument('--model_path', type=str, default='model_best.pth', help='the path of model')
    parser.add_argument('--episodes', type=int, default=200, help='how many episodes to train')
    parser.add_argument('--record_dir', type=str, default='./data_save/record.txt', help='the path to save the record')

    args = parser.parse_args()
    assert image_type in ['IR', 'visible'], 'image type must be IR or visible'
    if image_type == 'IR':
        args.train_dir = './IR_dataset/train_set'
        args.vali_dir = './IR_dataset/vali_set'
        args.model_path = './ir_model.pth'  # well_trained model is in the dir of ./model_dir
        args.image_size = [215, 161, 3]
    else:
        args.train_dir = './visible_dataset/train_set'
        args.vali_dir = './visible_dataset/vali_set'
        args.model_path = './visible_model.pth'  # well_trained model is in the dir of ./model_dir
        args.image_size = [480, 640, 3]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %% data part
    train_dataset = dataset_pt(args.image_size, args.target_size[0], dataset_path=args.train_dir, way=args.way,
                               shot=args.shot, query=args.query)
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_dataset = dataset_pt(args.image_size, args.target_size[0], dataset_path=args.vali_dir, way=args.way,
                             shot=args.shot,
                             query=args.query)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # %% model part
    net = VGG_embedding(input_shape=args.target_size, way=args.way, shot=args.shot, query=args.query, pretrained=False)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    criterion = EuclideanLoss(way=args.way, shot=args.shot, query=args.query)
    criterion.to(device)

    # %% train
    best_accuracy = 0
    for episode in range(args.episodes):
        print("train part")
        acc_train, ave_loss_train = train(net, train_loader, optimizer, criterion, episode=episode, device=device)
        print('validation part')
        acc_vali, ave_loss_vali = vali(net, val_loader, optimizer, criterion, episode=episode, device=device)
        f = open(args.record_dir, 'a')  # open file in append mode
        np.savetxt(f, np.c_[
            episode, acc_train, ave_loss_train, acc_vali, ave_loss_vali
        ])
        f.close()
        # %% early stop
        if acc_vali >= best_accuracy:
            best_accuracy = acc_vali
            torch.save(net.state_dict(),
                       args.model_path)
            print("best model saved,best_accuracy={:.6f}".format(best_accuracy))


if __name__ == '__main__':
    main_train(image_type='IR')
