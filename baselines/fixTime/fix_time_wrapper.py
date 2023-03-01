'''
@Author: WANG Maonan
@Date: 2023-03-01 16:50:51
@Description: 利用 Set Current Phase Duration 实现 Fix Time 的策略
@LastEditTime: 2023-03-01 17:28:25
'''
import logging
import gym
import numpy as np
from typing import List, Dict
from gym import spaces

class env_wrapper(gym.Wrapper):
    def __init__(
            self, 
            env, # 传入环境
            tls_id, 
            mode='train',
        ) -> None:
        super(env_wrapper, self).__init__(env)
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.tls_id = tls_id # 信号灯 id (原本是一个 multi-agent 的环境)
        # 训练或是测试模式, reset 有所区别
        self.mode = mode # train 模式正常
        self.reset_num = 0 # 第一次 reset 正常

        self.logger.debug(f'从 5-60 中选择数字')
        self._actions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # 所有可以选择的绿灯时间
        self.action_space = spaces.Discrete(12) # 动作空间
        

    def reset(self, ):
        """将 reset 返回的 obs 从 dict->list
        - 支持 reset 时候随机选择 route 文件和 net 文件
        - 提取一些 movement 的信息
        """
        if (self.mode == 'train') or (self.reset_num == 0):
            obs = self.env.reset()
        else:
            obs = 1
        
        self.reset_num += 1 # 重置次数
        return obs


    def step(self, action):
        """将 step 的结果提取, 从 dict 提取为 list

        Args:
            action (_type_): 设置当前相位的时间
        """
        action =  {self.tls_id: self._actions[action]}

        observations, rewards, dones, info = self.env.step(action)
        
        return observations, rewards, dones['__all__'], info