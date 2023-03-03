#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2022-08-22 21:40:18
 # @Description: 测试 eattention 在不同的数据增强方法的效果
 # @Command: nohup bash train_eattention.sh > train_eattention.log &
 # @LastEditTime: 2023-03-02 10:59:13
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

echo ${FOLDER}

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention
echo '完成 False, False, False, False.'

python ${FOLDER}/train.py --shuffle --laneNums --noise --mask --stack=6 --delay=0 --model_name=eattention
echo '完成 True, True, True, True.'
