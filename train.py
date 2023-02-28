'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 训练 RL 模型
@LastEditTime: 2023-02-24 23:20:10
'''
import os
from typing import Callable
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging

from SumoNets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG
from sumo_env import makeENV
from models import scnn

class VecNormalizeCallback(BaseCallback):
    """保存环境标准化之后的值
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super(VecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {path}")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)
    init_logging(log_path=pathConvert('./'), log_level=0)
    
    NUM_CPUS = 8
    EVAL_FREQ = 1000 # 一把交互 700 次
    SAVE_FREQ = EVAL_FREQ*2 # 保存的频率
    SHFFLE = True # 是否进行数据增强
    N_STACK = 4 # 堆叠
    N_DELAY = 0 # 时延
    MODEL_PATH = pathConvert(f'./models/{N_STACK}_{N_DELAY}_{SHFFLE}/')
    LOG_PATH = pathConvert('./log/') # 存放仿真过程的数据
    LOG_DIR = pathConvert('./tensorboard_logs/')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    # #########
    # 初始化环境
    # #########
    FOLDER_NAME = 'train_four_3' # 不同类型的路口
    tls_id = TRAIN_SUMO_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = TRAIN_SUMO_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    net_name = TRAIN_SUMO_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = TRAIN_SUMO_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = TRAIN_SUMO_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    # 转换为文件路径
    cfg_xml = pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{route_name}')]

    # 不同的 env 文件
    env_dict = {
        _folder: {
            'cfg': pathConvert(f'./SumoNets/{_folder}/env/{TRAIN_SUMO_CONFIG[_folder]["sumocfg"]}'),
            'net':[pathConvert(f'./SumoNets/{_folder}/env/{_net}') for _net in TRAIN_SUMO_CONFIG[_folder]['nets']],
            'route':[pathConvert(f'./SumoNets/{_folder}/routes/{_route}') for _route in TRAIN_SUMO_CONFIG[_folder]['routes']]
        }
        for _folder in ['train_four_3', 'train_four_345', 'train_three_3']
    }

    params = {
        'tls_id':tls_id,
        'begin_time':start_time,
        'num_seconds':3600,
        'sumo_cfg':cfg_xml,
        'net_files':net_xml,
        'route_files':route_xml,
        'is_shuffle':SHFFLE,
        'num_stack':N_STACK,
        'num_delayed':N_DELAY,
        'is_libsumo':True,
        'use_gui':False,
        'min_green':5,
        'log_file':LOG_PATH,
        'env_dict':env_dict
    }
    env = SubprocVecEnv([makeENV.make_env(env_index=f'{N_STACK}_{N_DELAY}_{SHFFLE}_{i}', **params) for i in range(NUM_CPUS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True) # 进行标准化
    
    # ########
    # callback
    # ########
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50,
        verbose=True
    ) # 何时停止
    eval_callback = EvalCallback(
        env,
        eval_freq=EVAL_FREQ,
        best_model_save_path=MODEL_PATH,
        callback_after_eval=stop_callback, # 每次验证之后需要调用
        verbose=1
    ) # 保存最优模型
    checkpoint_callback = CheckpointCallback(
        save_freq=3000,
        save_path=MODEL_PATH,
    ) # 定时保存模型
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=3000,
        save_path=MODEL_PATH,
    ) # 保存环境参数
    callback_list = CallbackList([eval_callback, checkpoint_callback, vec_normalize_callback])


    # ###########
    # start train
    # ###########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_kwargs = dict(
        features_extractor_class=scnn.SCNN,
        features_extractor_kwargs=dict(features_dim=32), # features_dim 提取的特征维数
    )
    model = PPO(
                "MlpPolicy", env, verbose=True, 
                policy_kwargs=policy_kwargs, learning_rate=linear_schedule(3e-4), 
                tensorboard_log=LOG_DIR, device=device
            )
    model.learn(total_timesteps=100000000, tb_log_name=f'{N_STACK}_{N_DELAY}_{SHFFLE}', callback=callback_list) # log 的名称

    # #########
    # save env
    # #########
    env.save(f'{MODEL_PATH}/vec_normalize.pkl')