'''
@Author: WANG Maonan
@Date: 2024-03-22 20:44:48
@Description: 基础 Wrapper, 处理 Obs, Acton, Reward
Noted: 这里和论文有一些不同, 我们将右转的 movement 也考虑在内, 进一步增强通用性
@LastEditTime: 2024-03-23 01:03:58
'''
import os
import numpy as np
import gymnasium as gym
from typing import Dict
from loguru import logger

class base_wrapper(gym.Wrapper):
    """
    A wrapper for a traffic signal control environment using reinforcement learning.
    """
    def __init__(self, env, root_folder:str, env_dict:Dict[str,str]=None) -> None:
        """
        Initialize the traffic signal control wrapper.

        :param env: The original gym environment to wrap.
        :param root_folder: The folder where network files are located.
        :param env_dict: A dictionary containing configurations for different environments.
        """
        super(base_wrapper, self).__init__(env)
        self.root_folder = root_folder # 路网所在的文件夹
        self.env_dict = env_dict # 存储文件信息
        
        # Initialize traffic signal ID and movement IDs to None; will be set in reset
        self.tls_id = None
        self.movement_ids = None # 记录每个 net 的 movement 顺序, [m1, m2, ...,]

        self.observation_space = gym.spaces.Box(
            low=0, 
            high=3,
            shape=(5,12,7)
        ) # obs 空间
        self.action_space = gym.spaces.Discrete(2) # 动作空间大小是 2， Keep or Change


    def reset(self, seed=None):
        """
        Reset the environment and return the initial observation.

        :param seed: An optional random seed to control randomness.
        :return: Initial observation after resetting the environment.
        """
        # Randomly choose an environment configuration
        env_id = np.random.choice(list(self.env_dict.keys()))
        env_config = self.env_dict[env_id]
        self.tls_id = env_config['tls_id']  # Set the traffic signal ID

        # Set up paths for the SUMO configuration, network, and route files
        sumo_cfg_path = os.path.join(self.root_folder, env_id, 'env', env_config['sumocfg'])
        net_file = np.random.choice(env_config['nets'])
        route_file = np.random.choice(env_config['routes'])

        # Log the chosen configuration
        logger.info(f'RL: Net: {net_file} || Route: {route_file}.')

        # Update the environment paths
        self.env.tsc_env._sumo_cfg = sumo_cfg_path
        self.env.tsc_env._net = os.path.join(self.root_folder, env_id, 'env', net_file)
        self.env.tsc_env._route = os.path.join(self.root_folder, env_id, 'routes', route_file)

        # Reset the environment and process the observation
        observations = self.env.reset()
        tls_state = observations['tls'][env_config['tls_id']]
        self.movement_ids = tls_state['movement_ids']
        processed_obs = self._process_obs(tls_state)
        
        return np.array([processed_obs]*5, dtype=np.float32), {}


    def step(self, action):
        """
        Execute an action in the environment and return the result.

        :param action: The action to perform.
        :return: A tuple containing the new observation, reward, and done flags.
        """
        # Execute the action until it can be performed
        can_perform_action = False
        total_obs = [] # max length=5
        total_reward = 0 # 将每个时刻的奖励求和

        while not can_perform_action:
            final_action = {self.tls_id: action} # 构建单路口 action 的动作
            _observations, rewards, truncated, dones, infos = self.env.step(final_action) # 达到最大仿真步就结束
            
            tls_state = _observations['tls'][self.tls_id]
            can_perform_action = tls_state['can_perform_action']
            
            # Process the observation and accumulate the reward
            processed_obs = self._process_obs(tls_state)
            total_reward += self._process_reward(_observations['vehicle'])

            # Ensure total_obs does not exceed length of 5
            if len(total_obs) >= 5:
                total_obs.pop(0)  # Remove the oldest observation
            total_obs.append(processed_obs)


        total_obs = np.array(total_obs, dtype=np.float32)
        return total_obs, total_reward, truncated, dones, infos
    

    def _process_obs(self, tls_state):
        """
        Process the traffic light state into a structured numpy array.

        :param tls_state: The raw state of the traffic light.
        :return: Processed observation as a numpy array.
            [occupancy, is_straight, is_left, is_right, lane_numbers, is_now_phase, is_next_phase]
        """    
        process_obs = []
        for _movement_index, _movement_id in enumerate(self.movement_ids):
            occupancy = tls_state['last_step_occupancy'][_movement_index]/100
            direction_flags = self._direction_to_flags(tls_state['movement_directions'][_movement_id])
            lane_numbers = tls_state['movement_lane_numbers'][_movement_id]/5 # 车道数 (默认不会超过 5 个车道)
            is_now_phase = int(tls_state['this_phase'][_movement_index])
            is_next_phase = int(tls_state['next_phase'][_movement_index])
            # 将其添加到 obs 中
            process_obs.append([occupancy, *direction_flags, lane_numbers, is_now_phase, is_next_phase])
            
        # 不是四岔路, 进行不全
        for _ in range(12 - len(process_obs)):
            process_obs.append([0]*len(process_obs[0]))

        return process_obs
    
    def _process_reward(self, vehicle_state):
        """
        Calculate the average waiting time for vehicles at the intersection.

        :param vehicle_state: The state of vehicles in the environment.
        :return: The negative average waiting time as the reward.
        """
        waiting_times = [veh['waiting_time'] for veh in vehicle_state.values()]
        return -np.mean(waiting_times) if waiting_times else 0

    def _direction_to_flags(self, direction):
        """
        Convert a direction string to a list of flags indicating the direction.

        :param direction: A string representing the direction (e.g., 's' for straight).
        :return: A list of flags for straight, left, and right.
        """
        return [
            1 if direction == 's' else 0,
            1 if direction == 'l' else 0,
            1 if direction == 'r' else 0
        ]