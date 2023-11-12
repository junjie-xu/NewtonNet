import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import *
from load_data import *
from synthetic import *
from utils import eval_rocauc, eval_acc



def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    # z = F.log_softmax(z, dim=1)
    loss = criterion(z[train_mask], true_label[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model):
    model.eval()
    z = model(data.x, data.edge_index)
    z = F.log_softmax(z, dim=1)

    train_acc = eval_func(data.y[train_mask].unsqueeze(1), z[train_mask])
    val_acc = eval_func(data.y[val_mask].unsqueeze(1), z[val_mask])
    test_acc = eval_func(data.y[test_mask].unsqueeze(1), z[test_mask])
    return train_acc, val_acc, test_acc



def parser_add_main_args(parser):
    # Data
    parser.add_argument('--dataname', type=str, default='Penn94')
    parser.add_argument('--num_masks', type=int, default=5, help='number of masks')
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')
    parser.add_argument('--test_prop', type=float, default=.2, help='test label proportion')

    # Model
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0., help='dropout for propagation layer')

    # Training
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=33780)

    # Case Study
    parser.add_argument('--mode', type=str, default='3', choices=['2', '3', '4'])
    parser.add_argument('--num_nodes', type=int, default=3000)
    parser.add_argument('--num_features', type=int, default=3000)
    parser.add_argument('--ratio', type=str, default='0.00', help='homophily ratio')
    parser.add_argument('--part1', type=float, default=1.0)
    parser.add_argument('--part2', type=float, default=1.0)
    parser.add_argument('--part3', type=float, default=1.0)
    parser.add_argument('--part4', type=float, default=1.0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    ratio_map = {
        '0.00': 0.00001, '0.05': 0.05,  '0.10': 0.10,  '0.15': 0.15,
        '0.20': 0.20, '0.25': 0.25, '0.30': 0.30, '0.35': 0.35,
        '0.40': 0.40, '0.45': 0.45, '0.50': 0.50, '0.55': 0.55,
        '0.60': 0.60, '0.65': 0.65, '0.70': 0.70, '0.75': 0.75,
        '0.80': 0.80, '0.85': 0.85, '0.90': 0.90, '0.95': 0.95, '1.00': 0.99999
    }
    ratio = ratio_map[args.ratio]
    data = CSBM(n=args.num_nodes, d=5, ratio=ratio, p=args.num_features, mu=1,
                train_prop=.025, valid_prop=.025, test_prop=0.95, num_masks=args.num_masks)
    print("Homophily Ratio: ", edge_homo_ratio(data))
    data = data.to(device)
    print('Args:', str(args))
    print('Data:', data)
    num_nodes = data.num_nodes
    num_classes = data.y.max().item() + 1
    num_features = data.x.shape[1]

    true_label = F.one_hot(data.y, data.y.max() + 1).squeeze(1).to(torch.float)

    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc


    # Training Process
    print("Training:")
    # Iterate each mask
    best_val_acc_multi, test_acc_multi = [], []
    for i in range(data.train_mask.shape[1]):
        print('MASK:', i)
        train_mask = data.train_mask[:, i]
        val_mask = data.val_mask[:, i]
        test_mask = data.test_mask[:, i]

        model = Filter(num_features, args.hidden, num_classes, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc = best_test_acc = 0
        for epoch in range(args.epochs + 1):
            train_loss = train(model, optimizer)

            train_acc, val_acc, test_acc = test(model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            if epoch % 200 == 0:
                log = 'Epoch:{:04d}, Train_loss:{:.4f}, Train_acc:{:.3f}, Val_acc:{:.3f}, Test_acc:{:.3f}, Best_test_acc:{:.3f}'
                print(log.format(epoch, train_loss, train_acc * 100, val_acc * 100, test_acc * 100, best_test_acc * 100))

        best_val_acc_multi.append(best_val_acc)
        test_acc_multi.append(best_test_acc)


    # Process results
    best_val_acc_multi.append(np.mean(best_val_acc_multi))
    test_acc_multi.append(np.mean(test_acc_multi))
    best_val_acc_multi = (np.array(best_val_acc_multi) * 100).reshape(-1,1)
    test_acc_multi = (np.array(test_acc_multi) * 100).reshape(-1, 1)
    result = np.around(np.concatenate((best_val_acc_multi, test_acc_multi), 1), decimals=3)
    print(result)
    print('Std.:{:.3f}'.format(np.std(result[0:args.num_masks, 1])))

    # Save results
    save_args = '_'.join(str(i) for i in [args.part1, args.part2, args.part3, args.part4,
                                          args.lr, args.dropout, args.weight_decay, args.hidden])
    path = os.path.join('result_case', args.mode, str(args.num_nodes)+'_'+str(args.num_features), args.ratio, save_args)
    file_name = path + '.csv'
    print(file_name)
    np.savetxt(file_name, result, fmt='%.03f', delimiter=',')
    print('\n\n')


