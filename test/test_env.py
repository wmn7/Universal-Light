'''
@Author: WANG Maonan
@Date: 2022-03-15 12:02:58
@Description: 测试环境 Next or Not
1. 测试 frame stack 和 obs delayed
2. 测试数据增强
@LastEditTime: 2022-06-21 22:05:00
'''
from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
from stable_baselines3.common.env_checker import check_env

pathConvert = getAbsPath(__file__)
import sys
sys.path.append(pathConvert('../'))


from sumo_env.makeENV import make_env
from SumoNets.net_config import SUMO_NET_CONFIG

if __name__ == '__main__':
    init_logging(log_path=pathConvert('./'), log_level=0)

    FOLDER_NAME = 'train_four_3' # 不同类型的路口
    tls_id = SUMO_NET_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = SUMO_NET_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    net_name = SUMO_NET_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = SUMO_NET_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = SUMO_NET_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    # 转换为文件路径
    cfg_xml = pathConvert(f'../SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'../SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'../SumoNets/{FOLDER_NAME}/routes/{route_name}')]

    # 不同的 env 文件
    env_dict = {
        _folder: {
            'cfg': pathConvert(f'../SumoNets/{_folder}/env/{SUMO_NET_CONFIG[_folder]["sumocfg"]}'),
            'net':[pathConvert(f'../SumoNets/{_folder}/env/{_net}') for _net in SUMO_NET_CONFIG[_folder]['nets']],
            'route':[pathConvert(f'../SumoNets/{_folder}/routes/{_route}') for _route in SUMO_NET_CONFIG[_folder]['routes']]
        }
        for _folder in ['train_four_3', 'train_four_345', 'train_three_3', 'train_three_34']
    }

    log_path = pathConvert('./log/')

    # 初始化环境
    g_envs = make_env(
        tls_id=tls_id,
        begin_time=start_time,
        num_seconds=3600,
        sumo_cfg=cfg_xml,
        net_files=net_xml,
        route_files=route_xml,
        env_dict=env_dict,
        num_stack=4, # 特征堆叠
        num_delayed=4, # obs 时延
        use_gui=False,
        min_green=5,
        log_file=log_path, # 存储 log 文件的路径
        env_index='test_env',
        is_shuffle=False
    )
    env1 = g_envs() # 生成新的环境
    check_env(env1, warn=True) # 测试环境

    # 随机执行动作, 查看 state, action, reward
    obs = env1.reset()
    done = False # 默认是 False
    
    # 打印相位结构
    print(env1.net_movements) # movement_id -> 转向
    print(env1.net_masks) # phase 包含哪些 movement
    print(env1.movement_info) # movement 包含的信息

    # 打印「obs space」和「action space」
    print(env1.observation_space)
    print(env1.action_space)

    while not done:
        action = env1.action_space.sample()
        obs, reward, done, info = env1.step(action) # 随机选择一个动作, 从 phase 中选择一个
        print(obs.shape) # 检查一下 obs 返回的值是否正确

    env1.close()
