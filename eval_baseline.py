'''
@Author: WANG Maonan
@Date: 2024-03-24 01:02:28
@Description: 简单实现 FixTime, 用于比较 waiting time
@LastEditTime: 2024-03-24 21:30:38
'''
import sys
from tshub.utils.get_abs_path import get_abs_path
pathConvert = get_abs_path(__file__)
sys.path.append(pathConvert('../../'))

from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from sumo_env.make_tsc_env import make_env # 创建环境
from sumo_datasets.TEST_CONFIG import TEST_SUMO_CONFIG as SUMO_CONFIG
# from sumo_datasets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG as SUMO_CONFIG

path_convert = get_abs_path(__file__)
logger.remove()

def init_env_config(config_name, config_data, net, route):
    """Initialize environment configuration for given parameters.
    """
    return {
        config_name: {
            'tls_id': config_data['tls_id'],
            'sumocfg': config_data['sumocfg'],
            'nets': [net],
            'routes': [route]
        }
    }

# Function to run the simulation
def run_simulation(env_config, config_name):
    """Run the simulation and return the total waiting time."""
    tsc_env_generate = make_env(
        root_folder=path_convert(f"./sumo_datasets/"),
        init_config=env_config[config_name],
        env_dict=env_config,
        num_seconds=3600,
        use_gui=False,
        log_file=path_convert('./log/'),
        env_index=0,
    )
    tsc_env = tsc_env_generate()

    total_reward = 0
    total_waiting_time = 0
    tsc_env.reset()
    dones = False
    while not dones:
        action = 0  # Change
        _, rewards, _, dones, infos = tsc_env.step(action=action)
        total_waiting_time += infos['avg_waiting_time']
        total_reward += rewards

    tsc_env.close()
    return total_reward, total_waiting_time

# Main loop to iterate over configurations and run simulations
for config_name, config_data in SUMO_CONFIG.items():
    for net in config_data['nets']:
        for route in config_data['routes']:
            # Initialize configuration for current net and route
            env_config = init_env_config(config_name, config_data, net, route)
            key = f"{config_name}||{net}||{route}"
            
            # Run the simulation
            total_reward, total_waiting_time = run_simulation(env_config, config_name)
            
            # Output the result
            print(f'{key} 的总等待时间为 {total_waiting_time:.2f} || Total Reward: {total_reward:.2f}.')