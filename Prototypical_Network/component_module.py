import torch
import torch.nn as nn
import torch.nn.functional as F

#%%

def train(net, dataloader, optimizer, criterion, episode,device):
    net.train()
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    total_correct = 0
    total_number = 0

    for epoch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        supports, queries = batch
        supports = supports.squeeze()
        queries = queries.squeeze()
        supports, queries = supports.to(device), queries.to(device)
        supports_feat, queries_feat = net(supports), net(queries, mode='query')
        dist, label = criterion(supports_feat, queries_feat)
        # prop=dist
        prop = F.softmax(dist)
        loss = ce_loss(prop, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = prop.argmax(dim=1)
        total_number += queries_feat.size(0)

        total_correct += (pred == label).sum().item()
    acc = total_correct / total_number
    print('Episode : {}, Loss : {}, accuracy : {}'.format(episode, total_loss / len(dataloader), acc))
    return acc, total_loss / len(dataloader)


def vali(net, dataloader, optimizer, criterion, episode,device):
    net.eval()
    total_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    total_correct = 0
    total_number = 0

    for epoch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        supports, queries = batch
        supports = supports.squeeze()
        queries = queries.squeeze()
        supports, queries = supports.to(device), queries.to(device)
        supports_feat, queries_feat = net(supports), net(queries, mode='query')
        dist, label = criterion(supports_feat, queries_feat)
        # prop=dist
        prop = F.softmax(dist)
        loss = ce_loss(prop, label)
        # loss = ce_loss(dist, label)
        total_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        pred = prop.argmax(dim=1)
        total_number += queries_feat.size(0)

        total_correct += (pred == label).sum().item()
    acc = total_correct / total_number
    print('Episode : {}, Loss : {}, accuracy : {}'.format(episode, total_loss / len(dataloader), acc))
    return acc, total_loss / len(dataloader)

class EuclideanLoss(nn.Module):

    def __init__(self, way, shot, query):
        super().__init__()
        self.way = way
        self.shot = shot
        self.query = query

    def forward(self, supports, queries):
        dist = -torch.cdist(supports, queries, p=2).T
        label = torch.LongTensor([i // self.query for i in range(queries.size(0))]).cuda()
        return dist, label
