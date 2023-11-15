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
import optuna


def train(model, optimizer, train_mask, ratio):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    # z = F.log_softmax(z, dim=1)
    class_loss = criterion(z[train_mask], true_label[train_mask])

    regu = -args.gamma * (ratio - face) * torch.norm(model.prop.temp[:onethird]) + \
            -args.gamma2 * (-torch.abs(ratio - face)) * torch.norm(model.prop.temp[onethird:twothirds]) + \
            -args.gamma3 * (face - ratio) * torch.norm(model.prop.temp[twothirds:])


    loss = class_loss + regu
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, train_mask, val_mask, test_mask):
    model.eval()
    z = model(data.x, data.edge_index)
    z = F.log_softmax(z, dim=1)

    train_acc = eval_func(data.y[train_mask].unsqueeze(1), z[train_mask])
    val_acc = eval_func(data.y[val_mask].unsqueeze(1), z[val_mask])
    test_acc = eval_func(data.y[test_mask].unsqueeze(1), z[test_mask])
    return train_acc, val_acc, test_acc


def work(args):
    best_val_acc_multi, test_acc_multi = [], []
    for i in range(data.train_mask.shape[1]):
        print('MASK:', i)
        train_mask = data.train_mask[:, i]
        val_mask = data.val_mask[:, i]
        test_mask = data.test_mask[:, i]

        ratio = torch.tensor(face)

        model = Newton(num_features, args.hidden, num_classes, args, points=torch.linspace(0, 2, steps=args.K+1)).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters()},
            {'params': model.lin2.parameters()},
            {'params': model.prop.parameters(), 'lr': args.temp_lr}
        ], lr=args.lr, weight_decay=args.weight_decay)


        best_val_acc = best_test_acc = 0
        for epoch in range(args.epochs + 1):
            train_loss = train(model, optimizer, train_mask, ratio)

            if epoch % 10 == 0:
                train_acc, val_acc, test_acc = test(model, train_mask, val_mask, test_mask)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

            if epoch % 10 == 0:
                z = model(data.x, data.edge_index)
                z = F.log_softmax(z, dim=1)
                pred = z.max(1)[1]
                ratio = homo_ratio(data, pred, train_mask)

            if epoch % 200 == 0:
                log = 'Epoch:{:04d}, Train_loss:{:.4f}, Train_acc:{:.3f}, Val_acc:{:.3f}, Test_acc:{:.3f}, Best_test_acc:{:.3f}'
                print(log.format(epoch, train_loss, train_acc * 100, val_acc * 100, test_acc * 100, best_test_acc * 100))


        best_val_acc_multi.append(best_val_acc)
        test_acc_multi.append(best_test_acc)
    return np.mean(best_val_acc_multi) * 100


def search_hyper_params(trial: optuna.Trial):
    lr = trial.suggest_categorical("lr", [0.05, 0.01, 0.005])
    temp_lr = trial.suggest_categorical("temp_lr", [0.05, 0.01, 0.005])
    dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
    dprate = trial.suggest_float("dprate", 0.0, 0.9, step=0.1)
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 5e-4])
    K = trial.suggest_categorical("K", [5])
    hidden = trial.suggest_categorical("hidden", [64])
    gamma = trial.suggest_categorical('gamma', [0.0, 1.0, 3.0, 5.0])
    gamma2 = trial.suggest_categorical('gamma2', [0.0, 1.0, 3.0, 5.0])
    gamma3 = trial.suggest_categorical('gamma3', [0.0, 1.0, 3.0, 5.0])
    
    args.lr = lr
    args.temp_lr = temp_lr
    args.dropout = dropout
    args.dprate = dprate
    args.weight_decay = weight_decay
    args.K = K
    args.hidden = hidden
    args.gamma = gamma
    args.gamma2 = gamma2
    args.gamma3 = gamma3
     
    return work(args)
    

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
    
    ratio = torch.tensor(face)

    true_label = F.one_hot(data.y, data.y.max() + 1).squeeze(1).to(torch.float)

    if args.dataname in ('twitch', 'fb100', 'Penn94', 'deezer',
                         'pokec', 'yelpchi', 'gamer', 'genius',):
        criterion = nn.BCEWithLogitsLoss()
        eval_func = eval_rocauc
    else:
        print("Not Binary!!!!")
    
    
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///" + 'newton_' + args.dataname + ".db",
                                study_name='newton_' + args.dataname,
                                )
    study.optimize(search_hyper_params, n_trials=args.trails)
    print("best params ", study.best_params)
    print("best valf1 ", study.best_value)


