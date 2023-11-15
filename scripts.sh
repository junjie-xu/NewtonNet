python main.py --dataname cora --lr 0.05 --temp_lr 0.01 --K 5 --hidden 64 --weight_decay 0.0005 --dropout 0.3 --dprate 0.5 --gamma 0 --gamma2 1 --gamma3 3

python main.py --dataname citeseer --lr 0.01 --temp_lr 0.05 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.0 --dprate 0.5 --gamma 0 --gamma2 3 --gamma3 5

python main.py --dataname pubmed --lr 0.05 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.3 --dprate 0.3 --gamma 0 --gamma2 0 --gamma3 1

python main.py --dataname chameleon --lr 0.05 --temp_lr 0.01 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.0 --dprate 0.0 --gamma 3 --gamma2 5 --gamma3 0

python main.py --dataname squirrel --lr 0.05 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.1 --dprate 0.5 --gamma 5 --gamma2 3 --gamma3 0

python main.py --dataname crocodile --lr 0.005 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0005 --dropout 0.0 --dprate 0.1 --gamma 3 --gamma2 3 --gamma3 0

python main.py --dataname texas --lr 0.05 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0005 --dropout 0.5 --dprate 0.0 --gamma 0 --gamma2 5 --gamma3 3

python main.py --dataname cornell --lr 0.05 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0005 --dropout 0.5 --dprate 0.1 --gamma 0 --gamma2 3 --gamma3 0

python main.py --dataname Penn94 --lr 0.005 --temp_lr 0.01 --K 5 --hidden 64 --weight_decay 0.0005 --dropout 0.3 --dprate 0.1 --gamma 0 --gamma2 1 --gamma3 0

python train_binary.py --dataname gamer --lr 0.005 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.5 --dprate 0.3 --gamma 1 --gamma2 0 --gamma3 0

python train_binary.py --dataname genius --lr 0.05 --temp_lr 0.005 --K 5 --hidden 64 --weight_decay 0.0 --dropout 0.0 --dprate 0.5 --gamma 0 --gamma2 1 --gamma3 0




