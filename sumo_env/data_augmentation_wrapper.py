'''
@Author: WANG Maonan
@Date: 2022-06-16 21:51:03
@Description: Including four data augmentation methods
- 针对路口信息
- - row shuffle
- - change lane num
- - flow scale
- 常用的数据增强
- - noise traffic
- - mask part movement info
@LastEditTime: 2024-03-25 08:35:42
'''
import logging
import numpy as np
import gymnasium as gym

class data_augmentation_wrapper(gym.ObservationWrapper):
    """
    """
    def __init__(
        self,
        env: gym.Env,
        is_shuffle: bool = True,
        is_change_lane: bool = False,
        is_flow_scale: bool = True, 
        is_noise: bool = False,
        is_mask: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)
        self.logger = logging.getLogger(__name__)
        self.is_shuffle = is_shuffle # 是否 shuffle
        self.is_change_lane = is_change_lane
        self.is_flow_scale = is_flow_scale
        self.is_noise = is_noise
        self.is_mask = is_mask
        self._phase_num = self.observation_space.shape[-2]
        assert self._phase_num == 12, '相位数是 12.'
        self._idx = list(range(self._phase_num))
        

    def _shuffle(self, observation):
        """对 obs 中每一个时刻进行打乱顺序。例如原始是：
            array([[[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2]],

                   [[3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5]],

                   [[6, 6, 6],
                    [7, 7, 7],
                    [8, 8, 8]]])

        转换之后变为：
            array([[[0, 0, 0],
                    [2, 2, 2],
                    [1, 1, 1]],

                   [[3, 3, 3],
                    [5, 5, 5],
                    [4, 4, 4]],

                   [[6, 6, 6],
                    [8, 8, 8],
                    [7, 7, 7]]])
        """
        self.logger.debug(f'Shuffle, {self.is_shuffle}, --> {self._idx}')
        
        # Apply the shuffle index to each 2D slice in the 3D array
        return observation[:, self._idx] # 进行乱序
    

    def _change_lane_num(self, observation):
        """对 observation 每行第 5 个元素 (index=4) 进行修改，也就是修改车道数
        车道数可以有 1,2,3,4,5 --> 0.2,0.4,0.6,0.8,1.0
        这里车道数是做了归一化, 例如 1 -> 0.2
        """
        _raw_lane_num = observation[0,:,4] # 原始车道数

        # Compute the ratio of new to old lane numbers, with zero division handling
        _ratio = np.divide(
            self._new_lane_num, _raw_lane_num, 
            out=np.zeros_like(self._new_lane_num), 
            where=_raw_lane_num!=0
        ) # 计算新旧车道的倍数

        # Apply the ratio to compute the final new lane numbers
        final_new_lane_num = _raw_lane_num*_ratio # 计算最终的车道数，使用乘法确保本来车道为 0 的还是车道为 0

        # 修改 obs 车道数
        observation[:,:,4] = final_new_lane_num

        return observation
    
    def _flow_scale(self, observation):
        """将 obs 的 flow 同时变大或是变小，乘上同一个数字, 希望 agent 关注相对数量, 而不是绝对数量
        """
        # Generate a random scaling factor
        _ratio = 0.8+0.4*np.random.rand() # noise range is 0.8-1.2

        # Apply the scaling factor to the first column of each 2D slice
        observation[:,:,0] *= _ratio

        return observation

    def _noise(self, observation):
        """对 obs 每行的前三个元素乘上一个随机数
        """
        if np.random.rand()>0.5:
            self.logger.debug(f'Add noise in traffic flow.')
            _noise = 0.9+0.2*np.random.rand((8))
            observation[:,:,0] = observation[:,:,0]*_noise
        return observation

    def _mask(self, observation):
        """对 obs 中某一片 movement info 遮住
        """
        if np.random.rand()>0.5:
            self.logger.debug(f'Add mask in movement info.')
            _slice = list(range(observation.shape[0]))
            _mask_index = np.random.choice(_slice) # 选择一个时间片
            observation[_mask_index,:,[5]] = 0 # 将 mask_index 的置 0
        return observation

    def observation(self, observation):
        obs_wrapper = observation[:]
        if self.is_shuffle:
            obs_wrapper = self._shuffle(obs_wrapper)
        if self.is_change_lane:
            obs_wrapper = self._change_lane_num(obs_wrapper)
        if self.is_flow_scale:
            obs_wrapper = self._flow_scale(obs_wrapper)
        if self.is_noise:
            obs_wrapper = self._noise(obs_wrapper)
        if self.is_mask:
            obs_wrapper = self._mask(obs_wrapper)
        return obs_wrapper


    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, done, and information from the environment
        """
        observation, reward, truncated, done, info = self.env.step(action)
        return self.observation(observation), reward, truncated, done, info


    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, _ = self.env.reset(**kwargs)

        if self.is_shuffle:
            # 一次仿真就 shuffle 一次就可以了
            np.random.shuffle(self._idx)
        
        if self.is_change_lane:
            self._new_lane_num = np.random.choice(
                np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32), 
                (self._phase_num,), 
                p=[0.4, 0.4, 0.1, 0.05, 0.05],
            ) # 给 12 个 movement 重新生成车道数

        return self.observation(obs), {}