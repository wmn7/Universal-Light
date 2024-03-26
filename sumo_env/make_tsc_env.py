'''
@Author: WANG Maonan
@Date: 2023-09-14 13:48:19
@Description: Create Traffic Signal Control Environment
@LastEditTime: 2024-03-24 21:34:20
'''
import gymnasium as gym
from typing import Dict, List
from stable_baselines3.common.monitor import Monitor

from .tsc_env import TSCEnvironment
from .base_wrapper import base_wrapper
from .data_augmentation_wrapper import data_augmentation_wrapper

def make_env(
        root_folder:str,
        init_config: Dict[str,str],
        env_dict: Dict[str, List],
        num_seconds:int, use_gui:bool,
        log_file:str, env_index:int,
        is_data_aug:bool = False
    ):
    def _init() -> gym.Env:
        init_tls_id,init_sumo_cfg = init_config['tls_id'], init_config['sumocfg']

        # 初始化环境
        tsc_scenario = TSCEnvironment(
            sumo_cfg=init_sumo_cfg, 
            num_seconds=num_seconds,
            tls_ids=[init_tls_id], 
            tls_action_type='next_or_not',
            use_gui=use_gui,
        )

        # 处理环境的 state, reward
        tsc_wrapper = base_wrapper(
            env=tsc_scenario, 
            root_folder=root_folder,
            env_dict=env_dict
        )

        # State Augmentation Methods
        if is_data_aug:
            tsc_wrapper = data_augmentation_wrapper(tsc_wrapper)
            
        return Monitor(tsc_wrapper, filename=f'{log_file}/{env_index}')
    
    return _init