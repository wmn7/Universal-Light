#!/usr/bin/zsh
###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:14:24
 # @Description: 测试不同的数据增强的组合，查看哪一个数据增强的方法是有效的 -> 测试 flowScale 和其他的搭配
 # @Command: nohup bash test_ad_flowScale.sh > test_ad_flowScale.log &
 # @LastEditTime: 2023-03-06 14:14:25
###
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

python ${FOLDER}/train.py --flowScale --stack=6 --delay=0 --model_name=ernn
echo '完成 flowScale.'

python ${FOLDER}/train.py --flowScale --noise --stack=6 --delay=0 --model_name=ernn
echo '完成 flowScale + Noise.'

python ${FOLDER}/train.py --flowScale --mask --stack=6 --delay=0 --model_name=ernn
echo '完成 flowScale + Mask.'