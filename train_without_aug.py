'''
@Author: WANG Maonan
@Date: 2024-03-23 01:06:18
@Description: 不使用数据增强进行训练（2phase 和 4phase 的奖励相差很大, 比较难以训练）
@LastEditTime: 2024-03-24 01:12:12
'''
import os
import torch
from loguru import logger
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from models.scnn import SCNN
from models.eattention import EAttention
from sumo_env.make_tsc_env import make_env
from utils.lr_schedule import linear_schedule
from SumoNets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG as SUMO_CONFIG # 训练路网的信息

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), log_level="INFO")

def create_env(params, CPU_NUMS=12):
    try:
        env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(CPU_NUMS)])
        return env
    except Exception as e:
        logger.error(f"Environment creation failed: {e}")
        raise

def train_model(env, tensorboard_path, callback_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_kwargs = dict(
        features_extractor_class=SCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )
    model = PPO(
        "MlpPolicy", 
        env, 
        batch_size=64,
        n_steps=1024, n_epochs=5,
        learning_rate=linear_schedule(3e-4),
        verbose=True, 
        policy_kwargs=policy_kwargs, 
        tensorboard_log=tensorboard_path, 
        device=device
    )
    model.learn(total_timesteps=1e6, tb_log_name='without_aug', callback=callback_list)
    return model


if __name__ == '__main__':
    log_path = path_convert('./log/')
    model_path = path_convert('./save_models/')
    tensorboard_path = path_convert('./tensorboard/')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    
    # Define the parameters for the environment creation
    FOLDER_NAME = 'train_four_3'
    params = {
        'root_folder': path_convert(f"./SumoNets/"),
        'init_config': {
            'tls_id': SUMO_CONFIG[FOLDER_NAME]['tls_id'],
            'sumocfg': path_convert(f"./SumoNets/{FOLDER_NAME}/env/{SUMO_CONFIG[FOLDER_NAME]['sumocfg']}")
        },
        'env_dict': SUMO_CONFIG,
        'num_seconds': 3600,
        'use_gui': False,
        'log_file': log_path,
        # Add any other parameters that are necessary for the environment
    }

    env = create_env(params)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=model_path)
    callback_list = CallbackList([checkpoint_callback])

    model = train_model(env, tensorboard_path, callback_list)

    # Save model and environment
    model.save(os.path.join(model_path, 'last_rl_model.zip'))
    logger.info('Training complete, reached maximum steps.')

    env.close()