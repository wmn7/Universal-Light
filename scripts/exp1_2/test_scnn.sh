#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 测试 scnn 模型在不同环境下的结果，测试的时候数据增强都是不开的，只会使用数据增强的模型，但是 state 不再加入数据增强
 # @Command: nohup bash test_scnn.sh > test_scnn.log &
 # @LastEditTime: 2023-03-06 14:14:25
### 
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

for net_name in train_four_3 train_four_345 train_three_3 test_four_34 test_three_34 cologne1 ingolstadt1
do
    python ${FOLDER}/test.py --stack=6 --delay=0 --model_name=scnn --net_name=$net_name
    echo 'Finish scnn False' $net_name

    python ${FOLDER}/test.py --shuffle --laneNums --flowScale --noise --mask --stack=6 --delay=0 --model_name=scnn --net_name=$net_name
    echo 'Finish scnn True' $net_name
done