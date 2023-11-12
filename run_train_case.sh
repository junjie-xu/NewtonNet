#! /bin/sh

for ratio in 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00
do
    for part1 in 0.0 0.4 0.8 1.2 1.6 2.0
    do
        for part2 in 0.0 0.4 0.8 1.2 1.6 2.0
        do
            for part3 in 0.0 0.4 0.8 1.2 1.6 2.0
            do
                python ./train_case.py \
                    --ratio $ratio \
                    --part1 $part1 \
                    --part2 $part2 \
                    --part3 $part3 \
                    --lr 0.05 \
                    --weight_decay 0.0005 \
                    --dropout 0.1 \
                    --dprate 0.1 \
                    --hidden 64 \
                    --epochs 1200 \
                    --train_prop 0.025 \
                    --valid_prop 0.025 \
                    --num_masks 5 \
                    --seed 724 \
                    --mode 3 
done
done
done
done
