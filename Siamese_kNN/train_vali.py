import argparse

import numpy as np
from torch.utils.data import DataLoader

from component_module import *
from dataset_batch import dataset_siam
from nets.siamese_new import Siamese
"""
train function"""

def main_train(image_type: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./IR_dataset/train_set', help='train dataset dir')
    parser.add_argument('--vali_dir', type=str, default='./IR_dataset/vali_set', help='test dataset dir')
    parser.add_argument('--datasave_dir', type=str, default='./data_save', help='where to store the processed data')
    parser.add_argument('--batch_size', type=int, default=15, help='half batch size')
    parser.add_argument('--image_size', type=list, default=[215, 161, 3], help='original image size')
    parser.add_argument('--target_size', type=list, default=[84, 84, 3], help='target image size')
    parser.add_argument('--model_path', type=str, default='./model_dir/ir_model.pth', help='model path')
    parser.add_argument('--epoch_size', type=int, default=200, help='epoch size')

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
    # %% data prepare
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    record_dir = './data_save/record.txt'
    # %% data part
    train_dataset = dataset_siam(args.image_size, args.target_size[0], dataset_path=args.train_dir,
                                 batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_dataset = dataset_siam(args.image_size, args.target_size[0], dataset_path=args.vali_dir,
                               batch_size=args.batch_size)
    vali_loader = DataLoader(val_dataset, batch_size=1)

    # %% model part
    net = Siamese(input_shape=args.target_size, pretrained=False)
    net.to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    # %% train part
    best_accuracy = 0
    for epoch in range(args.epoch_size):
        ave_loss_train, acc_train = train(net, loss, train_loader, epoch, device, optimizer)

        ave_loss_vali, acc_vali = vali(net, loss, vali_loader, epoch, device, optimizer)
        f = open(record_dir, 'a')  # open file in append mode
        np.savetxt(f, np.c_[
            epoch, acc_train, ave_loss_train, acc_vali, ave_loss_vali
        ])
        f.close()
        #  early stop-----------------
        if acc_vali >= best_accuracy:
            best_accuracy = acc_vali
            torch.save(net.state_dict(),
                       args.model_path)  # %((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
            print("best model saved,best_accuracy={:.6f}".format(best_accuracy))


if __name__ == '__main__':
    main_train('visible')
