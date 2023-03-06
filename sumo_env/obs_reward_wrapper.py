'''
@Author: WANG Maonan
@Date: 2022-06-16 21:51:03
@Description: 生成对应的 state, action, reward. 同时对 state 进行数据增强
@LastEditTime: 2022-08-10 20:33:13
'''
import gym
import numpy as np
from typing import List, Dict
from gym import spaces

class obs_reward_wrapper(gym.Wrapper):
    def __init__(
                self, 
                env, tls_id, mode='train',
                env_dict:Dict[str,str]=None, 
                is_movement:bool=False
            ) -> None:
        super(obs_reward_wrapper, self).__init__(env)
        self.env = env
        self.tls_id = tls_id # 信号灯 id
        self.env_dict = env_dict # 存储文件信息
        self.is_movement = is_movement # 特征按照 movement 还是 lane
        # 训练或是测试模式, reset 有所区别
        self.mode = mode # train 模式正常
        self.reset_num = 0 # 第一次 reset 正常（不管是 train 还是 eval）

        if self.is_movement:
            self.net_movements = dict() # 记录每个 net 的 movement 顺序, {'1': [m1, m2, ...], '2': [m1, m2, ..]}
            self.net_masks = dict() # 记录每个 net 的相位结构 {'net1': [[1,0,0,1],[0,1,1,0]], 'net2': [[],[]]}
            self.movement_info = dict() # 记录 movement 的信息, {'net1': {'m1':[方向, 车道数], }, 'net2': [[],[]]}

        self.observation_space = spaces.Box(
            low=0, 
            high=3,
            shape=(8,8)
        ) # obs 空间
        # 动作空间不需要重写定义, Next or Not 动作空间大小就是 2


    def reset(self):
        """将 reset 返回的 obs 从 dict->list
        """
        if (self.mode == 'train') or (self.reset_num == 0):
            _env_id = np.random.choice(list(self.env_dict.keys()))
            self.sumo_cfg = self.env_dict[_env_id]['cfg']
            self.net_files = self.env_dict[_env_id]['net']
            self.route_files = self.env_dict[_env_id]['route']

            # 随机选择 route 和 net
            if (self.route_files is not None) and (self.net_files is not None): # net 和 route 都不是 None
                # 随机一个 net 和 route 文件
                net_file = np.random.choice(self.net_files, 1)[0]
                route_file = np.random.choice(self.route_files, 1)[0]
                observations = self.env.reset(sumo_cfg=self.sumo_cfg, net_file=net_file, route_file=route_file)
            elif (self.route_files is not None) and (self.net_files is None): # route!=None, net=None
                route_file = np.random.choice(self.route_files, 1)[0]
                observations = self.env.reset(sumo_cfg=self.sumo_cfg, route_file=route_file)    
            elif (self.route_files is None) and (self.net_files is not None): # route=None, net!=None
                net_file = np.random.choice(self.net_files, 1)[0]
                observations = self.env.reset(sumo_cfg=self.sumo_cfg, net_file=net_file)                   
            else: # 不随机 route 文件
                observations = self.env.reset()

            _observations = observations[self.tls_id]

            # 初始化时, 初始化 net 的 movement 组合和 traffic light structure
            if (self.is_movement) and (self.env._net not in self.net_movements): # 初始化时提取每个 net 的 movement
                net_phase2movements = self.env.traffic_signals[self.tls_id].phase2movements

                _net_movement = list() # 存储每个 net 中 movement 的顺序
                for _, phase_movement_list in net_phase2movements.items():
                    for phase_movement in phase_movement_list: # 0: ['gsndj_s4--s', 'gsndj_n7--s', 'None--None', 'gsndj_s4--r', 'gsndj_n7--r']
                        direction = phase_movement.split('--')[1] # 获得方向
                        if direction not in ['None', 'r']: # 去除 右转 和 None
                            _net_movement.append(phase_movement)
                self.net_movements[self.env._net] = sorted(list(set(_net_movement))) # 需要排序, 确保 train 和 test 时候每个路网的 movement 是一样的

                _net_mask = list() # 存储 net mask
                for phase_index, phase_movement_list in net_phase2movements.items(): # {0: ['gsndj_s4--s', 'gsndj_n7--s', 'None--None', 'gsndj_s4--r', 'gsndj_n7--r']， 1: []}
                    _phase_mask = [0]*len(self.net_movements[self.env._net]) # 每个 phase 由哪些 movement 组成
                    for phase_movement in phase_movement_list: # ['gsndj_s4--s', 'gsndj_n7--s', 'None--None', 'gsndj_s4--r', 'gsndj_n7--r']
                        if phase_movement in self.net_movements[self.env._net]: # self.net_movements 是没有 右转 和 None
                            _phase_mask[self.net_movements[self.env._net].index(phase_movement)] = 1 # 对应位置转换为 1
                    _net_mask.append(_phase_mask.copy())
                self.net_masks[self.env._net] = np.array(_net_mask.copy(), dtype=np.int8)

                _movement_info = dict()
                for movement_id, movement_flow in _observations['flow'].items():
                    _direction = movement_id.split('--')[1] # 获得方向
                    if _direction not in ['None', 'r']: # 去除 右转 和 None
                        _is_s = 1 if _direction=='s' else 0 # 是否是直行, 1=s, 0=l
                        _lane_num = len(movement_flow) # 车道数
                        _movement_info[movement_id] = (_is_s, _lane_num) # 统计每个 movement 的「方向」和「车道数」
                self.movement_info[self.env._net] = _movement_info

            observation = self._process_obs(_observations)
        else:
            observation = np.zeros(self.observation_space.shape)
        
        self.reset_num += 1 # 重置次数
        return observation


    def step(self, action):
        """将 step 的结果提取, 从 dict 提取为 list
        """
        action = {self.tls_id: action}
        observations, rewards, dones, info = self.env.step(action)

        # 处理 obs
        _observations = observations[self.tls_id]
        observation = self._process_obs(_observations)

        # 处理 reward, 排队长度的「均值」和「方差」
        single_agent_reward = rewards[self.tls_id] # 单个 agent 的 reward
        process_reward = self._process_reward(single_agent_reward)
        return observation, process_reward, dones['__all__'], info


    def _process_reward(self, raw_reward:Dict[str, List[float]]):
        """对原始的 reward 进行处理, 这里计算的是所有 movement 的平均排队长度

        Args:
            raw_reward (Dict[str, List[float]]): 原始信息, 每个 movement 的排队车辆. 下面是示例数据:  
                {
                    '161701303#7.248--r': [0.0], 
                    '161701303#7.248--s': [0.0, 0.0], 
                    '161701303#7.248--l': [0.0], 
                    '29257863#2--r': [1.0], 
                    '29257863#2--s': [1.0, 1.0, 0.0], 
                    '29257863#2--l': [0.0, 0.0], 
                    'gsndj_n7--r': [0.0], 
                    'gsndj_n7--s': [0.2, 0.0], 
                    'gsndj_n7--l': [0.0], 
                    'gsndj_s4--r': [0.0], 
                    'gsndj_s4--s': [0.0, 0.0], 
                    'gsndj_s4--l': [0.0]
                }

        Returns:
            (float): 关于 reward 的计算, 首先计算所有 movement 排队的平均值, 接着使用 k-mean jam, K 为常数. 
            例如 K=2, 那么 jam 越大, 则 reward 越小.
        """
        tls_reward = list()
        for _, jam in raw_reward.items():
            tls_reward.extend(jam)
        
        return 2 - np.mean(tls_reward)


    def _process_obs(self, observation):
        """处理 observation, 将 dict 转换为 array.
        - 每个 movement 的 state 包含以下的部分, state 包含以下几个部分, 
            :[flow, mean_occupancy, max_occupancy, is_s, num_lane, mingreen, is_now_phase, is_next_phase]
        """
        phase_num = len(observation['phase_id']) # phase 的个数
        delta_time = observation['delta_time']
        phase_index = observation['phase_id'].index(1) # 相位所在 index
        next_phase_index = (phase_index+1)%phase_num
        phase_movements = self.net_masks[self.env._net][phase_index] # 得到一个 phase 有哪些 movement 组成的
        next_phase_movements = self.net_masks[self.env._net][next_phase_index]

        _observation_net_info = list() # 路网的信息
        for _movement_id, _movement in enumerate(self.net_movements[self.env._net]): # 按照 movment_id 提取
            flow = np.mean(observation['flow'][_movement])/delta_time # 假设每秒通过一辆车
            mean_occupancy = np.mean(observation['mean_occupancy'][_movement])/100
            max_occupancy = np.mean(observation['max_occupancy'][_movement])/100
            is_s = self.movement_info[self.env._net][_movement][0] # 是否是直行
            num_lane = self.movement_info[self.env._net][_movement][1]/5 # 车道数 (默认不会超过 5 个车道)
            is_now_phase = phase_movements[_movement_id] # now phase id
            min_green = observation['min_green'][0] if is_now_phase else 0 # min green
            is_next_phase = next_phase_movements[_movement_id] # next phase id
            
            _observation_net_info.append([flow, mean_occupancy, max_occupancy, is_s, num_lane, min_green, is_now_phase, is_next_phase])

        # 不是四岔路, 进行不全
        for _ in range(8 - len(_observation_net_info)):
            self.logger.debug(f'{self.env._net} 进行 obs 补全到 8.')
            _observation_net_info.append([0]*8)

        obs = np.array(_observation_net_info, dtype=np.float32) # 每个 movement 的信息
        return obs