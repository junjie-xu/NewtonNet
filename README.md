# NewtonNet: Shape-aware Graph Spectral Learning
An official PyTorch implementation of the paper [Shape-aware Graph Spectral Learning](https://openreview.net/pdf?id=Yui55YzCao).

In this repository, we implement the *NewtonNet with Shape-aware Regularization*.


## Contribution

+ We are the first to establish a well-defined relationship between graph frequencies and homophily ratios. We empirically and theoretically show that the more homophilous the graph is, the more beneficial the low-frequency is; while the more heterophilous the graph is, the more beneficial the high-frequency is. 
+ We propose a novel framework NewtonNet using Newton Interpolation with shape-aware regularization that can learn better filter encourages beneficial frequency and discourages harmful frequency, resulting in better node representations. 
+ Extensive experiments demonstrate the effectiveness of NewtonNet in various settings.


## Installation
Create a new virtual environment.
```
conda create --name NewtonNet python=3.10
```

Optinal 1: Install via requirements.txt
```
pip install -r requirements.txt
```

Optional 2: Install manually.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install gdown pandas
```

## Running Experiments

To reproduce the results in Table 1, use commands in scripts.sh
```
python main.py --dataname chameleon --lr 0.05 --temp_lr 0.01 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.0 --dprate 0.0 --gamma 3 --gamma2 5 --gamma3 0
```

To search your own optimal hyperparameters, use
```
python search_multi.py --dataname chameleon
python search_binary.py --dataname genius
```


To reproduce the results in Fig. 1, run
```
bash run_train_case.sh
```






