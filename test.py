'''
@Author: WANG Maonan
@Date: 2023-03-06 13:47:23
@Description: 测试不同的模型在不同环境下的结果
@LastEditTime: 2023-03-06 14:12:54
'''
import argparse
import shutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
pathConvert = getAbsPath(__file__)

from sumo_env import makeENV
from create_params import create_test_params

def test_model(
        exp_type, model_name, net_name, 
        is_shuffle, is_change_lane, is_flow_scale,
        is_noise, is_mask, 
        n_stack, n_delay,
        singleEnv=False, fineTune=False,
    ):
    if model_name == 'None':
        model_name = ''
    assert model_name in ['scnn', 'ernn', 'eattention', ''], f'Model name error, {model_name}'
    # args, 这里为了组合成模型的名字
    SHFFLE = is_shuffle # 是否进行数据增强
    CHANGE_LANE = is_change_lane
    FLOW_SCALE = is_flow_scale
    NOISE = is_noise
    MASK = is_mask
    N_STACK = n_stack # 堆叠
    N_DELAY = n_delay # 时延

    if fineTune:
        fineTune_s = '_fineTune_'
    else:
        fineTune_s = '_'

    if not singleEnv:
        MODEL_PATH = pathConvert(f'./results/{exp_type}/models/{model_name}/{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/best_model.zip')
        VEC_NORM = pathConvert(f'./results/{exp_type}/models/{model_name}/{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/best_vec_normalize.pkl')
        LOG_PATH = pathConvert(f'./results/{exp_type}/log/{model_name}/{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/') # 存放仿真过程的数据
        output_path = pathConvert(f'./results/{exp_type}/output/{model_name}/{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/{net_name}')
    else: # 如果是对单个环境训练的模型，模型名称中会有环境的名字
        MODEL_PATH = pathConvert(f'./results/{exp_type}/models/{model_name}/{net_name}{fineTune_s}{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/best_model.zip')
        VEC_NORM = pathConvert(f'./results/{exp_type}/models/{model_name}/{net_name}{fineTune_s}{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/best_vec_normalize.pkl')
        LOG_PATH = pathConvert(f'./results/{exp_type}/log/{model_name}/{net_name}{fineTune_s}{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/') # 存放仿真过程的数据
        output_path = pathConvert(f'./results/{exp_type}/output/{model_name}/{net_name}{fineTune_s}{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}/{net_name}')

    eval_params = create_test_params(
            net_name=net_name, output_folder=output_path,
            N_DELAY=N_DELAY, N_STACK=N_STACK, 
            LOG_PATH=LOG_PATH
        )
    for _key, eval_param in eval_params.items():
        # The environment for evaluating
        eval_env = SubprocVecEnv([makeENV.make_env(env_index=f'test_{N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}', **eval_param) for i in range(1)])
        eval_env = VecNormalize.load(load_path=VEC_NORM, venv=eval_env) # 进行标准化
        eval_env.training = False # 测试的时候不要更新
        eval_env.norm_reward = False

        # ###########
        # start train
        # ###########
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPO.load(MODEL_PATH, env=eval_env, device=device)

        # #########
        # 开始测试
        # #########
        obs = eval_env.reset()
        done = False # 默认是 False

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            # action = np.array([0]) # 对于 discrete 此时绿灯时间就是 5
            obs, reward, done, info = eval_env.step(action) # 随机选择一个动作, 从 phase 中选择一个
            
        eval_env.close()

        # 拷贝生成的 tls 文件
        _net, _route = _key.split('__')
        shutil.copytree(
            src=pathConvert(f'./SumoNets/{net_name}/add/'),
            dst=f'{output_path}/{_net}/{_route}/add/',
            ignore=shutil.ignore_patterns('*.add.xml'),
            dirs_exist_ok=True,
        )


if __name__ == '__main__':
    init_logging(log_path=pathConvert('./'), log_level=0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--laneNums', default=False, action='store_true')
    parser.add_argument('--flowScale', default=False, action='store_true')
    parser.add_argument('--noise', default=False, action='store_true')
    parser.add_argument('--mask', default=False, action='store_true')
    parser.add_argument('--stack', type=int, default=4)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='scnn')
    parser.add_argument('--exp_type', type=str, default='exp1_2')
    parser.add_argument('--net_name', type=str, default='test_four_34')
    parser.add_argument('--singleEnv', default=False, action='store_true')
    parser.add_argument('--fineTune', default=False, action='store_true')
    args = parser.parse_args()

    test_model(
        exp_type=args.exp_type,
        model_name=args.model_name, net_name=args.net_name,
        is_shuffle=args.shuffle, is_change_lane=args.laneNums, is_flow_scale=args.flowScale,
        is_mask=args.mask, is_noise=args.noise,
        n_stack=args.stack, n_delay=args.delay,
        singleEnv=args.singleEnv, fineTune=args.fineTune
    )