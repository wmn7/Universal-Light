#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试不同数据增强方法的效果
 # @Command: nohup bash data_augmentation.sh > data_augmentation.log &
 # @LastEditTime: 2023-03-02 10:59:13
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

echo ${FOLDER}

python ${FOLDER}/train.py --stack=4 --delay=0
echo '完成 False, False, False, False.'

python ${FOLDER}/train.py --shuffle --stack=4 --delay=0
echo '完成 True, False, False, False.'

python ${FOLDER}/train.py --laneNums --stack=4 --delay=0
echo '完成 False, True, False, False.'

python ${FOLDER}/train.py --noise --stack=4 --delay=0
echo '完成 False, False, True, False.'

python ${FOLDER}/train.py --mask --stack=4 --delay=0
echo '完成 False, False, False, True.'

python ${FOLDER}/train.py --shuffle --laneNums --noise --mask --stack=4 --delay=0
echo '完成 True, True, True, True.'
