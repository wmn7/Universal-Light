###
 # @Author: WANG Maonan
 # @Date: 2023-03-06 14:15:10
 # @Description: 测试不同的数据增强模型
 # @Command: nohup bash test_ad_models.sh > test_ad_models.log &
 # @LastEditTime: 2023-03-06 14:15:11
### 
FOLDER="/home/wmn/TrafficProject/AiolosZoo/Universal_Light"

for net_name in train_four_3 train_four_345 train_three_3 test_four_34 test_three_34 ingolstadt1
do
    # 1
    python ${FOLDER}/test.py --shuffle --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'shuffle' $net_name

    # 2
    python ${FOLDER}/test.py --shuffle --laneNums --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'shuffle, laneNums' $net_name

    # 3
    python ${FOLDER}/test.py --shuffle --flowScale --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'shuffle, flowScale' $net_name

    # 4
    python ${FOLDER}/test.py --shuffle --noise --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'shuffle, noise' $net_name

    # 5
    python ${FOLDER}/test.py --shuffle --mask --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'shuffle, mask' $net_name

    # 6
    python ${FOLDER}/test.py --laneNums --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'laneNums' $net_name

    # 7
    python ${FOLDER}/test.py --laneNums --flowScale --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'laneNums, flowScale' $net_name

    # 8
    python ${FOLDER}/test.py --laneNums --noise --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'laneNums, noise' $net_name

    # 9
    python ${FOLDER}/test.py --laneNums  -mask --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'laneNums, mask' $net_name

    # 10
    python ${FOLDER}/test.py --flowScale --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'flowScale' $net_name

    # 11
    python ${FOLDER}/test.py --flowScale --noise --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'flowScale, noise' $net_name

    # 12
    python ${FOLDER}/test.py --flowScale --mask --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'flowScale, mask' $net_name

    # 13
    python ${FOLDER}/test.py --noise --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'noise' $net_name

    # 14
    python ${FOLDER}/test.py --noise --mask --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'noise, mask' $net_name

    # 15
    python ${FOLDER}/test.py --mask --stack=6 --delay=0 --exp_type=exp3 --model_name=None --net_name=$net_name 
    echo 'mask' $net_name
done