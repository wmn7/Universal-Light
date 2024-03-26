'''
@Author: WANG Maonan
@Date: 2023-09-01 13:45:26
@Description: 给所有场景生成 1h 的 route
@LastEditTime: 2024-03-24 18:51:51
'''
import random
from loguru import logger
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# Step 1: Define the list of possible elements
elements = [
    [2, 16],
    [3, 15],
    [4, 14]
]

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/3phases.net.xml")

for i in range(5):
    # Step 2: Randomly choose one of the possible elements
    chosen_element = random.choice(elements)

    # Step 3: Generate a list of length 12 with random occurrences of the two chosen numbers
    result_list = [random.choice(chosen_element) for _ in range(60)]
    logger.info(f'选择的元素为: {result_list}.')

    generate_route(
        sumo_net=sumo_net,
        interval=[1]*60, # 仿真时间 60min
        edge_flow_per_minute={
            'E0': result_list,
            '-E1': result_list,
            '-E3': [18-j for j in result_list],
        }, # 每分钟每个 edge 有多少车
        edge_turndef={
            'E0__E1': [0.9]*60,
            '-E1__-E0': [0.9]*60,
        },
        veh_type={
            'ego': {'color':'26, 188, 156', 'probability':0.1},
            'background': {'color':'155, 89, 182', 'speed':15, 'probability':0.9},
        },
        output_trip=current_file_path('./testflow.trip.xml'),
        output_turndef=current_file_path('./testflow.turndefs.xml'),
        output_route=current_file_path(f'./routes/{i}.rou.xml'),
        interpolate_flow=False,
        interpolate_turndef=False,
    )