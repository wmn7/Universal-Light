'''
@Author: WANG Maonan
@Date: 2024-03-23 01:06:18
@Description: 不使用数据增强进行训练
@LastEditTime: 2024-03-24 21:55:27
'''
import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from models.scnn import SCNN
from models.eattention import EAttention
from sumo_env.make_tsc_env import make_env
from utils.lr_schedule import linear_schedule
from sumo_datasets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG as SUMO_CONFIG # 训练路网的信息

logger.remove()
path_convert = get_abs_path(__file__)
# set_logger(path_convert('./'), log_level="INFO")

def create_env(params, CPU_NUMS=12):
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(CPU_NUMS)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    return env

def train_model(env, tensorboard_path, callback_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_kwargs = dict(
        features_extractor_class=EAttention,
        features_extractor_kwargs=dict(features_dim=32),
    )
    model = PPO(
        "MlpPolicy", 
        env, 
        batch_size=128,
        n_steps=500, # 每次更新的样本数量为 n_steps*NUM_CPUS, n_steps 太小可能会收敛到局部最优
        n_epochs=5, # 每次更新时，用同一批数据进行优化的次数。
        learning_rate=linear_schedule(1e-3),
        verbose=True,
        policy_kwargs=policy_kwargs, 
        tensorboard_log=tensorboard_path, 
        device=device
    )
    model.learn(total_timesteps=2e6, tb_log_name='J1', callback=callback_list)
    return model


if __name__ == '__main__':
    IS_DATA_AUG = True # 是否使用数据增强
    
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
        'root_folder': path_convert(f"./sumo_datasets/"),
        'init_config': {
            'tls_id': SUMO_CONFIG[FOLDER_NAME]['tls_id'],
            'sumocfg': path_convert(f"./sumo_datasets/{FOLDER_NAME}/env/{SUMO_CONFIG[FOLDER_NAME]['sumocfg']}")
        },
        'env_dict': SUMO_CONFIG,
        'num_seconds': 3600,
        'use_gui': False,
        'log_file': log_path,
        'is_data_aug': IS_DATA_AUG
    }

    env = create_env(params)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=model_path
    )
    callback_list = CallbackList([checkpoint_callback])

    model = train_model(env, tensorboard_path, callback_list)

    # Save model and environment
    model.save(os.path.join(model_path, 'last_rl_model.zip'))
    logger.info('Training complete, reached maximum steps.')

    env.close()