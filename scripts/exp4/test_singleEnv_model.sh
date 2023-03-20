#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 测试在单个路网上训练的模型
 # @Command: nohup bash test_singleEnv_model.sh > test_singleEnv_model.log &
 # @LastEditTime: 2023-03-06 14:14:25
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

for net_name in test_four_34 test_three_34 ingolstadt1
do
    python ${FOLDER}/test.py --stack=6 --delay=0 --exp_type=exp4 --model_name=None --net_name=$net_name --singleEnv
    python ${FOLDER}/test.py --stack=6 --delay=0 --exp_type=exp4 --model_name=None --net_name=$net_name --singleEnv --fineTune
    python ${FOLDER}/test.py --stack=6 --delay=0 --shuffle --laneNums --flowScale --noise --mask --exp_type=exp4 --model_name=None --net_name=$net_name --singleEnv --fineTune
done