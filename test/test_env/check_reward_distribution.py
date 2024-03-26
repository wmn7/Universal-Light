'''
@Author: WANG Maonan
@Date: 2024-03-23 16:02:41
@Description: 检查不同环境中 reward 的大小分布
- 问题: 在实际中, 同一个 phase 包含越多的 movement, waiting time 会相对较小, 这是因为同一个时刻能够通行的方向比较多
- 这里我们将每个环境运行 3 次 (动作就是 0,1 顺序切换), 统计奖励的分布, 我们会统计两种奖励的方式
    - 单纯使用 waiting time
    - 将 waiting time/相位
@LastEditTime: 2024-03-24 18:57:16
'''
import sys
from tshub.utils.get_abs_path import get_abs_path
pathConvert = get_abs_path(__file__)
sys.path.append(pathConvert('../../'))

import json
from tshub.utils.get_abs_path import get_abs_path
from sumo_env.make_tsc_env import make_env # 创建环境
from sumo_datasets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG as SUMO_CONFIG

path_convert = get_abs_path(__file__)

env_rewards = {} # 记录环境总的奖励

# Iterate over each configuration in the SUMO_CONFIG dictionary
for config_name, config_data in SUMO_CONFIG.items():
    for net in config_data['nets']:
        for route in config_data['routes']:
            # Initialize a configuration dictionary for the current net and route
            _config = {}
            key = f"{config_name}||{net}||{route}"
            _config[config_name] = {
                'tls_id': config_data['tls_id'],
                'sumocfg': config_data['sumocfg'],
                'nets': [net,],
                'routes': [route,]
            }
            
            # Initialize the SUMO environment with the specific configuration
            tsc_env_generate = make_env(
                root_folder=path_convert(f"../../sumo_datasets/"),
                init_config=_config[config_name],
                env_dict=_config,
                num_seconds=3600,
                use_gui=False,
                log_file=path_convert('./log/'),
                env_index=0,
            )
            tsc_env = tsc_env_generate()

            # Start the simulation and record the total reward
            dones = False
            action = 0
            total_reward = 0 # 累积奖励
            tsc_env.reset()
            while not dones:
                action = (action + 1) % 2 # next or not
                states, rewards, truncated, dones, infos = tsc_env.step(action=action)
                total_reward += rewards

            tsc_env.close()
            
            env_rewards[key] = total_reward

# 将 dict 转换为 str
with open(path_convert("./normailze_wt.json"), "w") as f:
    json.dump(env_rewards, f, indent=4)