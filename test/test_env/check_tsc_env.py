'''
@Author: WANG Maonan
@Date: 2024-03-22 20:24:59
@Description: 检查环境是否可以正常运行
@LastEditTime: 2024-03-24 21:35:20
'''
import sys
from tshub.utils.get_abs_path import get_abs_path
pathConvert = get_abs_path(__file__)
sys.path.append(pathConvert('../../'))

import numpy as np
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from stable_baselines3.common.env_checker import check_env

from sumo_env.make_tsc_env import make_env # 创建环境
from sumo_datasets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG # SUMO 路网的设置

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))


if __name__ == '__main__':
    FOLDER_NAME = 'train_four_3' # 不同类型的路口
    tls_id = TRAIN_SUMO_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    sumo_cfg = TRAIN_SUMO_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config

    sumo_cfg = path_convert(f"../../sumo_datasets/{FOLDER_NAME}/env/{sumo_cfg}") # 完整的 SUMO 配置文件
    log_path = path_convert('./log/')

    init_config = {'tls_id': tls_id, 'sumocfg':sumo_cfg}
    tsc_env_generate = make_env(
        root_folder=path_convert(f"../../sumo_datasets/"),
        init_config=init_config,
        env_dict=TRAIN_SUMO_CONFIG,
        num_seconds=3600,
        use_gui=False,
        log_file=log_path,
        env_index=0,
        is_data_aug=True # 使用数据增强
    )
    tsc_env = tsc_env_generate()

    # Check Env
    print(tsc_env.observation_space.sample())
    print(tsc_env.action_space.n)
    check_env(tsc_env)

    # Simulation with environment
    dones = False
    tsc_env.reset()
    while not dones:
        action = np.random.randint(2) # next or not
        states, rewards, truncated, dones, infos = tsc_env.step(action=action)
        logger.info(f"SIM: {infos['step_time']} \n+State:\n{states}; \n+Reward:{rewards}.")
    tsc_env.close()