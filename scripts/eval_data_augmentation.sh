#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2022-08-22 21:40:18
 # @Description: 对数据增强进行消融实验
 # @LastEditTime: 2023-03-02 10:59:13
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

echo ${FOLDER}

python ${FOLDER}/train.py --stack=6 --delay=0 --model_name=eattention
echo '完成 False, False, False, False.'

python ${FOLDER}/train.py --shuffle --stack=6 --delay=0 --model_name=eattention
echo '完成 True, False, False, False.'

python ${FOLDER}/train.py --laneNums --stack=6 --delay=0 --model_name=eattention
echo '完成 False, True, False, False.'

python ${FOLDER}/train.py --noise --stack=6 --delay=0 --model_name=eattention
echo '完成 False, False, True, False.'

python ${FOLDER}/train.py --mask --stack=6 --delay=0 --model_name=eattention
echo '完成 False, False, False, True.'

python ${FOLDER}/train.py --shuffle --laneNums --noise --mask --stack=6 --delay=0 --model_name=eattention
echo '完成 True, True, True, True.'
