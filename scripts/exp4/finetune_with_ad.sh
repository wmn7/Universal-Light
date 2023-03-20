#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 加载预训练模型进行微调，预训练模型是有数据增强
 # @Command: nohup bash finetune_with_ad.sh > finetune_with_ad.log &
 # @LastEditTime: 2023-03-06 14:14:25
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"
CPU=8

for net_name in test_four_34 test_three_34 ingolstadt1
do
    python ${FOLDER}/fine_tune.py --net_name=$net_name --shuffle --laneNums --flowScale --noise --mask --stack=6 --cpus=$CPU --delay=0
    echo '完成, ' $net_name
done