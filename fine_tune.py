'''
@Author: WANG Maonan
@Date: 2022-06-17 16:15:04
@Description: 将泛化模型在特定的场景上进行微调, 与直接在该场景训练的结果做对比
@LastEditTime: 2022-07-11 20:37:16
'''
import argparse
import os

import torch
from aiolos.trafficLog.initLog import init_logging
from aiolos.utils.get_abs_path import getAbsPath
from create_params import create_singleEnv_params
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sumo_env import makeENV
from utils.env_normalize import VecBestNormalizeCallback, VecNormalizeCallback
from utils.lr_schedule import linear_schedule

def fine_tune(
        net_name, 
        is_shuffle, is_change_lane, is_flow_scale,
        is_noise, is_mask, 
        n_stack, n_delay, num_cpus
    ):
    """微调的时候关闭全部的数据增强
    但是可以加载有数据增强的预训练模型，和没有数据增强的预训练模型
    """
    SHFFLE = is_shuffle # 是否进行数据增强
    CHANGE_LANE = is_change_lane
    FLOW_SCALE = is_flow_scale
    NOISE = is_noise
    MASK = is_mask
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延

    NUM_CPUS = num_cpus
    EVAL_FREQ = 2000 # 一把交互 700 次
    SAVE_FREQ = EVAL_FREQ*2 # 保存的频率
    MODEL_PATH = pathConvert(f'./results/models/{net_name}_fineTune_{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/')
    LOG_PATH = pathConvert(f'./results/log/{net_name}_fineTune_{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/') # 存放仿真过程的数据
    TENSORBOARD_LOG_DIR = pathConvert(f'./results/tensorboard_logs/{net_name}_fineTune/')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    # 所有的环境不需要数据增强
    train_params = create_singleEnv_params(
        net_name=net_name, 
        is_shuffle=False, is_change_lane=False, is_flow_scale=False,
        is_mask=False, is_noise=False, 
        N_DELAY=N_DELAY, N_STACK=N_STACK, 
        LOG_PATH=LOG_PATH
    )
    eval_params = create_singleEnv_params(
        net_name=net_name, 
        is_shuffle=False, is_change_lane=False, is_flow_scale=False,
        is_mask=False, is_noise=False, 
        N_DELAY=N_DELAY, N_STACK=N_STACK, 
        LOG_PATH=LOG_PATH
    )

    # The environment for training
    env = SubprocVecEnv([makeENV.make_env(env_index=f'{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}_{i}', **train_params) for i in range(NUM_CPUS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True) # 进行标准化
    # The environment for evaluating
    eval_env = SubprocVecEnv([makeENV.make_env(env_index=f'evaluate_{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}', **eval_params) for i in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True) # 进行标准化
    eval_env.training = False # 测试的时候不要更新
    eval_env.norm_reward = False

    # 加载预训练模型和环境参数
    PRETRAIN_MODEL_FOLDER = pathConvert(f'./results/exp3/models/6_0_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/')
    PRETRAIN_MODEL_PATH = os.path.join(PRETRAIN_MODEL_FOLDER, 'best_model.zip')
    PRETRAIN_VEC_NORM = os.path.join(PRETRAIN_MODEL_FOLDER, 'best_vec_normalize.pkl')

    # 加载环境 Norm 参数
    env = VecNormalize.load(load_path=PRETRAIN_VEC_NORM, venv=env)
    eval_env = VecNormalize.load(load_path=PRETRAIN_VEC_NORM, venv=eval_env)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPO.load(PRETRAIN_MODEL_PATH, env=env, learning_rate=linear_schedule(3e-4), device=device)

    # ########
    # callback
    # ########
    save_vec_normalize = VecBestNormalizeCallback(save_freq=1, save_path=MODEL_PATH)
    eval_callback = EvalCallback(
        eval_env, # 这里换成 eval env 会更加稳定
        eval_freq=EVAL_FREQ,
        best_model_save_path=MODEL_PATH,
        callback_on_new_best=save_vec_normalize,
        verbose=1
    ) # 保存最优模型
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_PATH,
    ) # 定时保存模型
    vec_normalize_callback = VecNormalizeCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_PATH,
    ) # 保存环境参数
    callback_list = CallbackList([eval_callback, checkpoint_callback, vec_normalize_callback])

    model.learn(
        total_timesteps=2e5, 
        tensorboard_log=TENSORBOARD_LOG_DIR,
        tb_log_name=f'fineTune_{net_name}_{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}', 
        callback=callback_list
    ) # log 的名称


if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)
    init_logging(log_path=pathConvert('./'), log_level=0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', type=str, default='test_four_34')
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--laneNums', default=False, action='store_true')
    parser.add_argument('--flowScale', default=False, action='store_true')
    parser.add_argument('--noise', default=False, action='store_true')
    parser.add_argument('--mask', default=False, action='store_true')
    parser.add_argument('--stack', type=int, default=4)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=8) # 同时开启的仿真数量
    args = parser.parse_args()

    fine_tune(
        net_name=args.net_name,
        is_shuffle=args.shuffle, is_change_lane=args.laneNums, is_flow_scale=args.flowScale,
        is_mask=args.mask, is_noise=args.noise,
        n_stack=args.stack, n_delay=args.delay,
        num_cpus=args.cpus
    )