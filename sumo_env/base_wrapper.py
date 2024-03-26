'''
@Author: WANG Maonan
@Date: 2024-03-22 20:44:48
@Description: 基础 Wrapper, 处理 Obs, Acton, Reward
Noted: 这里和论文有一些不同, 我们将右转的 movement 也考虑在内, 进一步增强通用性
@LastEditTime: 2024-03-24 18:55:21
'''
import os
import numpy as np
import gymnasium as gym
from typing import Dict
from loguru import logger

from .reward_utils import FixedRangeNormalizer, RewardNormalizer

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
        self.num_phases = None # 相位的数量
        self.num_lanes = None # 车道的数量
        self.movement_ids = None # 记录每个 net 的 movement 顺序, [m1, m2, ...,]
        self.reward_normalizer = {} # 为每一个环境写一个 normalizer

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
        self.env_id = np.random.choice(list(self.env_dict.keys()))
        env_config = self.env_dict[self.env_id]
        self.tls_id = env_config['tls_id']  # Set the traffic signal ID

        # Set up paths for the SUMO configuration, network, and route files
        sumo_cfg_path = os.path.join(self.root_folder, self.env_id, 'env', env_config['sumocfg'])
        self.net_file = np.random.choice(env_config['nets'])
        route_file = np.random.choice(env_config['routes'])

        # Log the chosen configuration
        logger.info(f'RL: Net: {self.net_file} || Route: {route_file}.')

        # Update the environment paths
        self.env.tsc_env._sumo_cfg = sumo_cfg_path
        self.env.tsc_env._net = os.path.join(self.root_folder, self.env_id, 'env', self.net_file)
        self.env.tsc_env._route = os.path.join(self.root_folder, self.env_id, 'routes', route_file)

        # Init Reward Normalize
        self.key = f'{self.env_id}__{self.net_file}'
        if self.key not in self.reward_normalizer:
            self.reward_normalizer[self.key] = RewardNormalizer()

        # Reset the environment and process the observation
        observations = self.env.reset()
        tls_state = observations['tls'][env_config['tls_id']]
        self.movement_ids = tls_state['movement_ids']
        self.num_lanes = sum([_num if _id.split('--')[1] != 'r' else 0 for _id, _num in tls_state['movement_lane_numbers'].items()]) # 车道数量
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
        total_rewards = [] # 将每个时刻的奖励求平均值
        self.info_avg_waiting_time = [] # 直接记录所有车辆的等待时间, 通过 info 传出去, 用于比较结果

        while not can_perform_action:
            final_action = {self.tls_id: action} # 构建单路口 action 的动作
            _observations, rewards, truncated, dones, infos = self.env.step(final_action) # 达到最大仿真步就结束
            
            tls_state = _observations['tls'][self.tls_id]
            can_perform_action = tls_state['can_perform_action']
            
            # Process the observation and accumulate the reward
            processed_obs = self._process_obs(tls_state)
            total_rewards.append(self._process_normalized_reward(_observations['vehicle'])) # 每一个时刻的 reward
            # total_rewards.append(self._process_reward(_observations['vehicle']))

            # Ensure total_obs does not exceed length of 5
            if len(total_obs) >= 5:
                total_obs.pop(0)  # Remove the oldest observation
            total_obs.append(processed_obs)

        total_obs = np.array(total_obs, dtype=np.float32)

        # Normalize Reward
        total_reward = sum(total_rewards)/len(total_rewards) # 计算每个时刻的平均奖励
        infos['avg_waiting_time'] = np.mean(self.info_avg_waiting_time) # 在 info 里面记录一下平均等待时间, 方便后期比较
        self.reward_normalizer[self.key].update(total_reward) # 需要为每一个 env 维护一个归一化参数
        final_reward = self.reward_normalizer[self.key].normalize(total_reward)
        final_reward = total_reward
        return total_obs, final_reward, truncated, dones, infos
    

    def _process_obs(self, tls_state):
        """
        Process the traffic light state into a structured numpy array.

        :param tls_state: The raw state of the traffic light.
        :return: Processed observation as a numpy array.
            [occupancy, is_straight, is_left, is_right, lane_numbers, is_now_phase, is_next_phase]
        """
        self.num_open_lanes_during_green = 0 # 计算绿灯阶段有多少 lane 可以通行

        process_obs = []
        for _movement_index, _movement_id in enumerate(self.movement_ids):
            occupancy = tls_state['last_step_occupancy'][_movement_index]/100
            direction_flags = self._direction_to_flags(tls_state['movement_directions'][_movement_id])
            lane_numbers = tls_state['movement_lane_numbers'][_movement_id]/5 # 车道数 (默认不会超过 5 个车道)
            is_now_phase = int(tls_state['this_phase'][_movement_index])
            is_next_phase = int(tls_state['next_phase'][_movement_index])
            # 将其添加到 obs 中
            process_obs.append([occupancy, *direction_flags, lane_numbers, is_now_phase, is_next_phase]) # 某个 movement 对应的信息

            if is_now_phase:
                self.num_open_lanes_during_green += tls_state['movement_lane_numbers'][_movement_id]
            
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
        waiting_times = [veh['waiting_time'] for veh in vehicle_state.values()] # 这里需要优化所有车的等待时间, 而不是停止车辆的等待时间
        self.info_avg_waiting_time.append(np.mean(waiting_times) if waiting_times else 0)
        
        return -np.mean(waiting_times) if waiting_times else 0

    def _process_normalized_reward(self, vehicle_state, max_waiting_time=60):
        """
        Calculate a normalized reward based on vehicle waiting times, number of phases, and number of lanes.
        Clip the waiting time to a maximum value.

        :param vehicle_state: The state of vehicles in the environment.
        :param max_waiting_time: The maximum allowed waiting time for a vehicle.
        :return: The normalized reward.
        """
        # Clip waiting times to the maximum value and filter out vehicles with zero waiting time
        waiting_times = [min(veh['waiting_time'], max_waiting_time) for veh in vehicle_state.values()]
        total_waiting_time = sum(waiting_times)
        num_vehicles = len(waiting_times) # 只统计大于 0 的车的数量
        
        # green_phase_efficiency = self.num_open_lanes_during_green / self.num_lanes # 绿灯时可以通行的车道占比
        green_phase_efficiency = 1 # 直接使用 Fix Norm, 不使用这个来优化了

        # Normalize by the number of vehicles and the efficiency of green phases
        if num_vehicles > 0 and green_phase_efficiency > 0:
            normalized_waiting_time = total_waiting_time / (num_vehicles * green_phase_efficiency)
        else:
            normalized_waiting_time = 0
        self.info_avg_waiting_time.append(normalized_waiting_time) # 添加变换之前的 waiting time

        # Invert the normalized waiting time as we want to reward lower waiting times
        # and penalize higher waiting times
        reward = 1 / normalized_waiting_time if normalized_waiting_time > 0 else 1
        
        return reward
    
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