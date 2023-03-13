#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 测试 ernn 模型在不同环境下的结果
 # @Command: nohup bash test_ernn.sh > test_ernn.log &
 # @LastEditTime: 2023-03-06 14:14:25
### 
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

for net_name in train_four_3 train_four_345 train_three_3 test_four_34 test_three_34 cologne1 ingolstadt1
do
    python ${FOLDER}/test.py --stack=6 --delay=0 --model_name=ernn --net_name=$net_name
    echo 'Finish ernn False' $net_name

    python ${FOLDER}/test.py --shuffle --laneNums --flowScale --noise --mask --stack=6 --delay=0 --model_name=ernn --net_name=$net_name
    echo 'Finish ernn True' $net_name
done