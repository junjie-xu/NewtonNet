from utils import rand_train_test_idx, index_to_mask, WikipediaNetwork2, even_quantile_labels
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor
import scipy.io
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import torch_geometric.transforms as T
import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import pandas as pd
import json
from ogb.nodeproppred import NodePropPredDataset
from os import path
import gdown
from torch_sparse import SparseTensor
# from google_drive_downloader import GoogleDriveDownloader as gdd
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def load_dataset(dataname, train_prop, valid_prop, test_prop, num_masks):
    assert dataname in ('cora', 'citeseer', 'pubmed', 'texas', 'wisconsin', 'cornell', 'squirrel',
                        'chameleon', 'crocodile', 'computers', 'photo', 'actor', 'twitch', 'fb100',
                        'Penn94', 'deezer', 'year', 'snap-patents', 'pokec', 'yelpchi', 'gamer',
                        'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'genius'), 'Invalid dataset'

    if dataname in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='./data/', name=dataname)
        data = dataset[0]
        # data.train_mask = torch.unsqueeze(data.train_mask, dim=1)
        # data.val_mask = torch.unsqueeze(data.val_mask, dim=1)
        # data.test_mask = torch.unsqueeze(data.test_mask, dim=1)

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root='./data/', name=dataname)
        data = dataset[0]

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    # elif dataname in ['squirrel', 'chameleon']:
    #     dataset = WikipediaNetwork(root='./data/', name=dataname, geom_gcn_preprocess=True)
    #     data = dataset[0]

    elif dataname in ['squirrel', 'chameleon']:
        preProcDs = WikipediaNetwork(root='./data/', name=dataname, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(root='./data/', name=dataname, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['crocodile']:
        dataset = WikipediaNetwork2(root='./data/', name=dataname, geom_gcn_preprocess=False)
        data = dataset[0]
        data.y = data.y.long()

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['computers', 'photo']:
        dataset = Amazon(root='./data/', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'Penn94':
        data = load_fb100_dataset(dataname, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop, num_masks=num_masks)

    elif dataname == 'twitch':
        data = load_twitch_dataset('DE', train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop, num_masks=num_masks)

    elif dataname == 'fb100':
        data = load_fb100_dataset(' ', train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop, num_masks=num_masks)

    elif dataname == 'deezer':
        data = load_deezer()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'genius':
        data = load_genius()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'year':
        data = load_arxiv_year()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'snap-patents':
        data = load_snap_patents()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'pokec':
        data = load_pokec()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'yelpchi':
        data = load_yelpchi()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'gamer':
        data = load_twitch_gamer_dataset()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    # elif dataname in ('ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'):
    #     dataset = PygNodePropPredDataset(name=dataname)
    #     data = dataset[0]
    #     data.y = data.y.squeeze()
    #     splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
    #                   for _ in range(num_masks)]
    #     data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    return data


dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}
DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'


def load_deezer():
    # filename = 'deezer-europe'
    # dataset = NCDataset(filename)
    deezer = scipy.io.loadmat('./data/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()

    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=label.shape[0])
    return data


def load_genius():
    filename = 'genius'
    # dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=label.shape[0])
    return data


def load_fb100_dataset(sub_dataname, train_prop, valid_prop, test_prop, num_masks):
    assert sub_dataname in ('Amherst41', 'Cornell5', 'Johns Hopkins55', 'Penn94', 'Reed98'), 'Invalid dataset'
    A, metadata = load_fb100(sub_dataname)
    # dataset = NCDataset(filename)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    metadata = metadata.astype(np.int64)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    label = torch.tensor(label, dtype=torch.long)

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]

    # if sub_dataname == 'Penn94':
    #     splits_lst = np.load('./data/splits/fb100-Penn94-splits.npy', allow_pickle=True)
    # else:
    splits_lst = [rand_train_test_idx(label, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
    train_mask, val_mask, test_mask = index_to_mask(splits_lst, num_nodes)

    data = Data(x=node_feat, edge_index=edge_index, y=label,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

    return data


def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat('./data/facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata


def load_twitch_dataset(sub_dataname, train_prop, valid_prop, test_prop, num_masks):
    assert sub_dataname in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(sub_dataname)
    # dataset = NCDataset(lang)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    num_nodes = node_feat.shape[0]

    # if sub_dataname == 'DE':
    #     splits_lst = np.load('./data/splits/twitch-e-DE-splits.npy', allow_pickle=True)
    # else:
    splits_lst = [rand_train_test_idx(label, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                         for _ in range(num_masks)]
    train_mask, val_mask, test_mask = index_to_mask(splits_lst, num_nodes)

    data = Data(x=node_feat, edge_index=edge_index, y=label,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

    return data


def load_twitch(sub_dataname):
    assert sub_dataname in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = './data/twitch/' + sub_dataname
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{sub_dataname}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{sub_dataname}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{sub_dataname}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0]  # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features


def load_arxiv_year(nclass=5):
    # filename = 'arxiv-year'
    # dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    graph = ogb_dataset.graph
    edge_index = torch.as_tensor(graph['edge_index'])
    x = torch.as_tensor(graph['node_feat'])

    label = even_quantile_labels(graph['node_year'].flatten(), nclass, verbose=False)
    y = torch.as_tensor(label).reshape(-1)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def load_snap_patents(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
                       output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    # dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_pokec():
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
                       output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    # dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_yelpchi():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
            output=f'{DATAPATH}YelpChi.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    # dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
                       output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
                       output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)

    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)

    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

    # dataset = NCDataset("twitch-gamer")
    # dataset.graph = {'edge_index': edge_index,
    #                  'node_feat': node_feat,
    #                  'edge_feat': None,
    #                  'num_nodes': num_nodes}
    label = torch.tensor(label)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features

