###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:15:10
 # @Description: 测试 attention 模型
 # @Command: nohup bash test_eattention.sh > test_eattention.log &
 # @LastEditTime: 2023-03-06 14:15:11
### 
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

for net_name in train_four_3 train_four_345 train_three_3 test_four_34 test_three_34 cologne1 ingolstadt1
do
    python ${FOLDER}/test.py --stack=6 --delay=0 --model_name=eattention --net_name=$net_name
    echo 'Finish eattention False' $net_name

    python ${FOLDER}/test.py --shuffle --laneNums --noise --mask --stack=6 --delay=0 --model_name=eattention --net_name=$net_name
    echo 'Finish eattention True' $net_name
done