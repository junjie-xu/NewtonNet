import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import *
from load_data import *
from config import *
from synthetic import *
from utils import eval_rocauc, eval_acc, homo_ratio



def train(model, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    z = F.log_softmax(z, dim=1)
    class_loss = F.nll_loss(z[train_mask], data.y[train_mask])

    regu = -args.gamma * (ratio - face) * torch.norm(model.prop.temp[:onethird]) + \
            -args.gamma2 * (-torch.abs(ratio - face)) * torch.norm(model.prop.temp[onethird:twothirds]) + \
            -args.gamma3 * (face - ratio) * torch.norm(model.prop.temp[twothirds:])

    loss = class_loss + regu
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model):
    model.eval()
    z = model(data.x, data.edge_index)
    z = F.log_softmax(z, dim=1)

    train_pred = z[train_mask].max(1)[1]
    train_acc = train_pred.eq(data.y[train_mask]).sum().item() / train_mask.sum().item()
    val_pred = z[val_mask].max(1)[1]
    val_acc = val_pred.eq(data.y[val_mask]).sum().item() / val_mask.sum().item()
    test_pred = z[test_mask].max(1)[1]
    test_acc = test_pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
    return train_acc, val_acc, test_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load Data
    data = load_dataset(args.dataname, args.train_prop, args.valid_prop, args.test_prop, args.num_masks)
    data = data.to(device)
    print('Args:', str(args))
    print('Data:', data)
    num_nodes = data.num_nodes
    num_classes = data.y.max().item() + 1
    num_features = data.x.shape[1]
    onethird, twothirds = int((args.K + 1) / 3), int(2 * (args.K + 1) / 3)
    half = int((args.K + 1) / 2)
    face = 1 / num_classes


    # Training Process
    print("Training:")
    # Iterate each mask
    best_val_acc_multi, test_acc_multi = [], []
    for i in range(data.train_mask.shape[1]):
        print('MASK:', i)
        train_mask = data.train_mask[:, i]
        val_mask = data.val_mask[:, i]
        test_mask = data.test_mask[:, i]

        ratio = torch.tensor(face)

        model = Newton(num_features, args.hidden, num_classes, args, points=torch.linspace(0, 2, steps=args.K+1)).to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters()},
            {'params': model.lin2.parameters()},
            {'params': model.prop.parameters(), 'lr': args.temp_lr}
        ], lr=args.lr, weight_decay=args.weight_decay)


        best_val_acc = best_test_acc = 0
        for epoch in range(args.epochs + 1):
            train_loss = train(model, optimizer, epoch)

            # if epoch % 20 == 0:
            train_acc, val_acc, test_acc = test(model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc


            if epoch % 10 == 0:
                z = model(data.x, data.edge_index)
                z = F.log_softmax(z, dim=1)
                pred = z.max(1)[1]
                ratio = homo_ratio(data, pred, train_mask)


            if epoch % 100 == 0:
                log = 'Epoch:{:04d}, Train_loss:{:.4f}, Train_acc:{:.3f}, Val_acc:{:.3f}, Test_acc:{:.3f}, Best_test_acc:{:.3f}'
                print(log.format(epoch, train_loss, train_acc * 100, val_acc * 100, test_acc * 100, best_test_acc * 100))
                # print('Ratio: ', ratio)


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
    if args.save_results:
        save_args = '_'.join(str(i) for i in [args.lr, args.temp_lr, args.K, args.gamma, args.gamma2, args.gamma3,
                                            args.dropout, args.dprate, args.hidden, args.weight_decay])
        path = os.path.join('result', args.dataname, 'gamma', save_args)
        file_name = path + '.csv'
        print(file_name)
        np.savetxt(file_name, result, fmt='%.03f', delimiter=',')
        print('\n')


