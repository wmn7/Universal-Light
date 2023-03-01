'''
@Author: WANG Maonan
@Date: 2023-03-01 18:18:16
@Description: 使用 SOTL 验证路口
@LastEditTime: 2023-03-01 18:50:38
'''
from aiolos.utils.get_abs_path import getAbsPath
pathConvert = getAbsPath(__file__)
import os
import sys
import shutil
sys.path.append(pathConvert('../'))

from SOTL.make_sotl_env import make_env
from SumoNets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG as SUMO_CONFIG

if __name__ == '__main__':
    FOLDER_NAME = 'train_four_3' # 不同类型的路口
    tls_id = SUMO_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = SUMO_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    net_name = SUMO_CONFIG[FOLDER_NAME]['nets'][1] # network file
    route_name = SUMO_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = SUMO_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    # 转换为文件路径
    cfg_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/env/{net_name}')
    route_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/routes/{route_name}')
    # output 统计文件
    _net = net_name.split('.')[0]
    _route = route_name.split('.')[0]
    output_folder = pathConvert(f'./Result/SOTL/{FOLDER_NAME}/{_net}/{_route}/')
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
        use_gui=True,
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
        if obs > 0.1:
            action = 1
        else:
            action = 0
        obs, reward, done, info = env1.step(action) # 随机选择一个动作, 从 phase 中选择一个

    env1.close()

    # 拷贝生成的 tls 文件
    shutil.copytree(
        src=pathConvert(f'../SumoNets/{FOLDER_NAME}/add/'),
        dst=f'{output_folder}/add/',
        ignore=shutil.ignore_patterns('*.add.xml'),
        dirs_exist_ok=True,
    )