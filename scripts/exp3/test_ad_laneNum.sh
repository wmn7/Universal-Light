#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 测试不同的数据增强的组合，查看哪一个数据增强的方法是有效的 -> 测试 laneNum 和其他的搭配
 # @Command: nohup bash test_ad_laneNum.sh > test_ad_laneNum.log &
 # @LastEditTime: 2023-03-06 14:14:25
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

python ${FOLDER}/train.py --laneNums --stack=6 --delay=0 --model_name=ernn
echo '完成 LaneNums.'

python ${FOLDER}/train.py --laneNums --flowScale --stack=6 --delay=0 --model_name=ernn
echo '完成 laneNums + FlowScale.'

python ${FOLDER}/train.py --laneNums --noise --stack=6 --delay=0 --model_name=ernn
echo '完成 laneNums + Noise.'

python ${FOLDER}/train.py --laneNums --mask --stack=6 --delay=0 --model_name=ernn
echo '完成 laneNums + Mask.'