'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 训练 RL 模型
@LastEditTime: 2023-02-24 23:20:10
'''
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging

from sumo_env import makeENV
from models import scnn
from create_params import create_params
from utils.lr_schedule import linear_schedule
from utils.env_normalize import VecNormalizeCallback, VecBestNormalizeCallback

if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)
    init_logging(log_path=pathConvert('./'), log_level=0)
    
    NUM_CPUS = 8
    EVAL_FREQ = 2000 # 一把交互 700 次
    SAVE_FREQ = EVAL_FREQ*2 # 保存的频率
    SHFFLE = True # 是否进行数据增强
    N_STACK = 4 # 堆叠
    N_DELAY = 0 # 时延
    MODEL_PATH = pathConvert(f'./results/models/{N_STACK}_{N_DELAY}_{SHFFLE}/')
    LOG_PATH = pathConvert(f'./results/log/{N_STACK}_{N_DELAY}_{SHFFLE}/') # 存放仿真过程的数据
    TENSORBOARD_LOG_DIR = pathConvert('./results/tensorboard_logs/')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    train_params = create_params(is_eval=False, SHFFLE=SHFFLE, N_DELAY=N_DELAY, N_STACK=N_STACK, LOG_PATH=LOG_PATH)
    eval_params = create_params(is_eval=True, SHFFLE=SHFFLE, N_DELAY=N_DELAY, N_STACK=N_STACK, LOG_PATH=LOG_PATH)
    # The environment for training
    env = SubprocVecEnv([makeENV.make_env(env_index=f'{N_STACK}_{N_DELAY}_{SHFFLE}_{i}', **train_params) for i in range(NUM_CPUS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True) # 进行标准化
    # The environment for evaluating
    eval_env = SubprocVecEnv([makeENV.make_env(env_index=f'evaluate_{N_STACK}_{N_DELAY}_{SHFFLE}_{i}', **eval_params) for i in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True) # 进行标准化
    eval_env.training = False # 测试的时候不要更新
    eval_env.norm_reward = False

    # ########
    # callback
    # ########
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5000,
        verbose=True
    ) # 何时停止
    save_vec_normalize = VecBestNormalizeCallback(save_freq=1, save_path=MODEL_PATH)
    eval_callback = EvalCallback(
        eval_env, # 这里换成 eval env 会更加稳定
        eval_freq=EVAL_FREQ,
        best_model_save_path=MODEL_PATH,
        callback_on_new_best=save_vec_normalize,
        callback_after_eval=stop_callback, # 每次验证之后需要调用
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
                tensorboard_log=TENSORBOARD_LOG_DIR, device=device
            )
    model.learn(total_timesteps=5e6, tb_log_name=f'{N_STACK}_{N_DELAY}_{SHFFLE}', callback=callback_list) # log 的名称

    # #########
    # save env
    # #########
    env.save(f'{MODEL_PATH}/vec_normalize.pkl')