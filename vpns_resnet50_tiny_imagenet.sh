#!/bin/sh
experiment_name='vpns'
foler_name=logs/${experiment_name}
if [ ! -d ${foler_name} ]; then
    mkdir -p ${foler_name}
fi
# network: resnet18 resnet50 vgg16
network='resnet50'
# dataset: cifar100 cifar10 oxfordpets dtd food101 flowers102 stanfordcars tiny_imagenet
dataset='tiny_imagenet'
prune_method='vpns'
# seed: 7 9 17
seeds=(7 9 17)
gpus=(0 0 0)

epochs=30
density_list='1,0.6,0.5,0.4,0.3,0.2,0.1'
weight_optimizer='sgd'
weight_lr=0.01
weight_vp_optimizer=${weight_optimizer}
weight_vp_lr=${weight_lr}
score_optimizer='adam'
score_lr=0.0001
score_vp_optimizer=${score_optimizer}
score_vp_lr=${score_lr}
for m in ${!seeds[@]};do            
    log_filename=${foler_name}/${network}_${dataset}_${prune_method}_${seeds[m]}_${weight_optimizer}_${weight_lr}_${weight_vp_optimizer}_${weight_vp_lr}_${score_optimizer}_${score_lr}_${score_vp_optimizer}_${score_vp_lr}.log
    nohup python -u main.py \
        --experiment_name ${experiment_name} \
        --dataset ${dataset} \
        --network ${network} \
        --prune_method ${prune_method} \
        --density_list ${density_list} \
        --weight_optimizer ${weight_optimizer} \
        --weight_lr ${weight_lr} \
        --weight_vp_optimizer ${weight_vp_optimizer} \
        --weight_vp_lr ${weight_vp_lr} \
        --score_optimizer ${score_optimizer} \
        --score_lr ${score_lr} \
        --score_vp_optimizer ${score_vp_optimizer} \
        --score_vp_lr ${score_vp_lr} \
        --gpu ${gpus[m]} \
        --epochs ${epochs} \
        --seed ${seeds[m]} \
        > $log_filename 2>&1 &
done
