'''
@Author: WANG Maonan
@Date: 2023-03-01 16:58:14
@Description: 测试 Fix Time 策略的结果, 最终输出的结果在 Result
@LastEditTime: 2023-03-06 13:21:52
'''
from aiolos.utils.get_abs_path import getAbsPath
pathConvert = getAbsPath(__file__)
import os
import sys
import shutil
sys.path.append(pathConvert('../'))

from fixTime.make_fix_time_env import make_env
from SumoNets.NET_CONFIG import SUMO_NET_CONFIG as SUMO_CONFIG

def sim_fix_time(net_name:str, fix_time:int):
    """测试 net_name 下的仿真场景，其中测试时间是 fix time
    """
    fix_time_map = {
        20:3, 25:4,
        30:5, 35:6,
        40:7, 45:8,
    }
    assert fix_time in fix_time_map, f'fix_time 只能是 20-45.'

    FOLDER_NAME = net_name # 不同类型的路口
    tls_id = SUMO_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = SUMO_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    nets_name = SUMO_CONFIG[FOLDER_NAME]['nets'] # network file (包含多个路网文件)
    routes_name = SUMO_CONFIG[FOLDER_NAME]['routes'] # route file (包含多个车辆文件)
    start_time = SUMO_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    for net_name in nets_name:
        for route_name in routes_name:
            # 转换为文件路径
            cfg_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/env/{cfg_name}')
            net_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/env/{net_name}')
            route_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/routes/{route_name}')
            # output 统计文件
            _net = net_name.split('.')[0]
            _route = route_name.split('.')[0] # 将 1.rou.xml 得到 1
            output_folder = pathConvert(f'./Result/FixTime/{FOLDER_NAME}/{fix_time}/{_net}/{_route}/')
            os.makedirs(output_folder, exist_ok=True) # 创建文件夹
            trip_info = os.path.join(output_folder, f'tripinfo.out.xml')
            statistic_output = os.path.join(output_folder, f'statistic.out.xml')
            summary = os.path.join(output_folder, f'summary.out.xml')
            queue_output = os.path.join(output_folder, f'queue.out.xml')
            tls_add = [
                # 探测器
                pathConvert(f'../SumoNets/{FOLDER_NAME}/detectors/e1_internal.add.xml'),
                pathConvert(f'../SumoNets/{FOLDER_NAME}/detectors/e2.add.xml'),
                # 信号灯
                pathConvert(f'../SumoNets/{FOLDER_NAME}/add/tls_programs.add.xml'),
                pathConvert(f'../SumoNets/{FOLDER_NAME}/add/tls_state.add.xml'),
                pathConvert(f'../SumoNets/{FOLDER_NAME}/add/tls_switch_states.add.xml'),
                pathConvert(f'../SumoNets/{FOLDER_NAME}/add/tls_switches.add.xml')
            ]

            # 初始化环境
            g_envs = make_env(
                tls_id=tls_id,
                begin_time=start_time,
                num_seconds=3600,
                sumo_cfg=cfg_xml,
                net_file=net_xml,
                route_file=route_xml,
                use_gui=False,
                min_green=5,
                trip_info=trip_info,
                statistic_output=statistic_output,
                summary=summary,
                queue_output=queue_output,
                tls_state_add= tls_add,
                mode='eval'
            )
            env1 = g_envs() # 生成新的环境
            obs = env1.reset()
            done = False # 默认是 False

            while not done:
                action = fix_time_map[fix_time] # action: 5->30; 7 -> 40
                obs, reward, done, info = env1.step(action) # 随机选择一个动作, 从 phase 中选择一个

            env1.close()

            # 拷贝生成的 tls 文件
            shutil.copytree(
                src=pathConvert(f'../SumoNets/{FOLDER_NAME}/add/'),
                dst=f'{output_folder}/add/',
                ignore=shutil.ignore_patterns('*.add.xml'),
                dirs_exist_ok=True,
            )


if __name__ == '__main__':
    test_nets = [
        'train_four_3', 'train_four_345', 'train_three_3', 
        'test_four_34', 'test_three_34', 'cologne1', 'ingolstadt1'
    ]
    for _fix_time in [20, 30, 40]:
        for _net in test_nets:
            sim_fix_time(net_name=_net, fix_time=_fix_time)