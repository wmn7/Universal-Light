#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 使用带有数据增强的模型来训练单个路网
 # @Command: nohup bash train_with_ad.sh > train_with_ad.log &
 # @LastEditTime: 2023-03-06 14:14:25
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"
CPU=8

for net_name in test_four_34 test_three_34 ingolstadt1
do
    python ${FOLDER}/train_singleEnv.py --net_name=$net_name --shuffle --laneNums --flowScale --noise --mask --stack=6 --cpus=$CPU --delay=0 --model_name=ernn
    echo '完成, ' $net_name
done