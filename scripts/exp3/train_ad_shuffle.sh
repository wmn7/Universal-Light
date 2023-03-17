#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 测试不同的数据增强的组合，查看哪一个数据增强的方法是有效的 -> 测试 shuffle 和其他的搭配
 # @Command: nohup bash test_ad_shuffle.sh > test_ad_shuffle.log &
 # @LastEditTime: 2023-03-06 14:14:25
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

python ${FOLDER}/train.py --shuffle --stack=6 --delay=0 --model_name=ernn
echo '完成 Shuffle.'

python ${FOLDER}/train.py --shuffle --laneNums --stack=6 --delay=0 --model_name=ernn
echo '完成 Shuffle + LaneNum.'

python ${FOLDER}/train.py --shuffle --flowScale --stack=6 --delay=0 --model_name=ernn
echo '完成 Shuffle + FlowScale.'

python ${FOLDER}/train.py --shuffle --noise --stack=6 --delay=0 --model_name=ernn
echo '完成 Shuffle + Noise.'

python ${FOLDER}/train.py --shuffle --mask --stack=6 --delay=0 --model_name=ernn
echo '完成 Shuffle + Mask.'