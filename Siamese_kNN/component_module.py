import heapq
import torch
import torch.nn as nn


def train(network, loss_func, dataloader, epoch,device, optimizer):
    network.train()
    total_loss = 0
    total_accuracy = 0

    for iteration, batch in enumerate(dataloader):
        left, right, label = batch
        left = left.squeeze()
        right = right.squeeze()
        label = label.squeeze(0)
        left, right, targets = left.to(device), right.to(device), label.to(device)
        optimizer.zero_grad()
        x, x1, x2 = network(left, right)
        outputs = nn.Sigmoid()(x)

        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            equal = torch.eq(torch.round(outputs), targets)
            accuracy = torch.mean(equal.float())
        total_loss += loss.item()
        total_accuracy += accuracy.item()

    iteration_sum = iteration + 1
    print('Train,Episode : {}, Loss : {}, accuracy : {}'.format(epoch, total_loss / iteration_sum,
                                                                total_accuracy / iteration_sum))
    return total_loss / iteration_sum, total_accuracy / iteration_sum


def vali(network, loss_func, dataloader, epoch,device, optimizer):
    network.eval()  # open vali mode
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader):
            left, right, label = batch
            left = left.squeeze()
            right = right.squeeze()
            label = label.squeeze(0)
            left, right, targets = left.to(device), right.to(device), label.to(device)
            optimizer.zero_grad()
            x, x1, x2 = network(left, right)
            outputs = nn.Sigmoid()(x)
            loss = loss_func(outputs, targets)
            # loss.backward()
            # optimizer.step()
            equal = torch.eq(torch.round(outputs), targets)
            accuracy = torch.mean(equal.float())
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            # print('Episode : {}, Loss : {}, accuracy : {}'.format(iteration, output.item(), accuracy.item()))
    iteration_sum = iteration + 1
    print('Validation, Episode : {}, Loss : {}, accuracy : {}'.format(epoch, total_loss / iteration_sum,
                                                                      total_accuracy / iteration_sum))
    return total_loss / iteration_sum, total_accuracy / iteration_sum


def test_for_label(net, train_loader, right,device):
    net.eval()
    with torch.no_grad():
        for train_index, train_data in enumerate(train_loader):
            image_train, label_train, train_image_dir = train_data
            left = image_train

            left, right = left.to(device), right.to(device)

            x, support_embedding, test_embedding = net(left, right)
            outputs = nn.Sigmoid()(x)
            outputs = outputs.to('cpu')
            # print('stop here')
            if train_index == 0:
                label_list = label_train
                prob_list = outputs
                support_set_emb = support_embedding
            else:
                label_list = torch.cat((label_list, label_train), dim=0)
                prob_list = torch.cat((prob_list, outputs), dim=0)
                support_set_emb = torch.cat((support_set_emb, support_embedding), dim=0)
        prob_list = prob_list.squeeze().numpy()
        max_values = heapq.nlargest(7, range(len(prob_list)), prob_list.take)  # select 7
        pred_lables_max_5 = [label_list[max_values[i]] for i in range(len(max_values))]
        maxlabel = max(pred_lables_max_5, key=pred_lables_max_5.count)
        predict_label = maxlabel

    return predict_label, support_set_emb, test_embedding
