'''
@Author: WANG Maonan
@Date: 2022-06-16 21:51:03
@Description: Including four data augmentation methods
- 针对路口信息
- - row shuffle
- - change lane num
- 常用的数据增强
- - noise traffic
- - mask part movement info
@LastEditTime: 2022-08-10 20:33:13
'''
import logging
import gym
import numpy as np

class data_augmentation_wrapper(gym.ObservationWrapper):
    """
    """
    def __init__(
        self,
        env: gym.Env,
        is_shuffle: bool,
        is_change_lane: bool,
        is_flow_scale: bool, 
        is_noise: bool,
        is_mask: bool,
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
        _phase_num = self.observation_space.shape[-2]
        assert _phase_num == 8, '相位数是 8.'
        self._idx = list(range(_phase_num))
        

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
        np.random.shuffle(self._idx)
        self.logger.debug(f'Shuffle, {self.is_shuffle}, --> {self._idx}')
        return observation[:, self._idx] # 进行乱序
    

    def _change_lane_num(self, observation):
        """对 observation 每行第四个元素进行修改，也就是修改车道数
        车道数可以有 1,2,3,4,5 --> 0.2,0.4,0.6,0.8,1.0
        """
        if np.random.rand()>0.5:
            self.logger.debug(f'Change Lane Number')
            _raw_lane_num = observation[0,:,4] # 原始车道数
            _new_lane_num = np.random.choice(
                np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32), 
                (8,), p=[0.25, 0.4, 0.2, 0.1, 0.05],
            ) # 给 8 个 movement 重新生成车道数
            _ratio = np.divide(_new_lane_num, _raw_lane_num, out=np.zeros_like(_new_lane_num), where=_raw_lane_num!=0) # 计算新旧车道的倍数
            final_new_lane_num = _raw_lane_num*_ratio # 计算最终的车道数，使用乘法确保本来车道为 0 的还是车道为 0

            # 修改 obs 车道数
            observation[:,:,4] = final_new_lane_num
            # 修改对应的流量, 不需要修改流量, 因为计算到每个车道的平均值
            # observation[:,:,0] = observation[:,:,0]*_ratio
            # observation[:,:,1] = observation[:,:,1]*_ratio
            # observation[:,:,2] = observation[:,:,2]*_ratio

        return observation
    
    def _flow_scale(self, observation):
        """将 obs 的 flow 同时变大或是变小，乘上同一个数字
        """
        if np.random.rand()>0.5:
            _ratio = 0.8+0.7*np.random.rand() # noise 的范围是 0.8 - 1.5
            observation[:,:,0] = observation[:,:,0]*_ratio
            observation[:,:,1] = observation[:,:,1]*_ratio
            observation[:,:,2] = observation[:,:,2]*_ratio
        return observation

    def _noise(self, observation):
        """对 obs 每行的前三个元素乘上一个随机数
        """
        if np.random.rand()>0.5:
            self.logger.debug(f'Add noise in traffic flow.')
            _noise = 0.9+0.2*np.random.rand((8))
            observation[:,:,0] = observation[:,:,0]*_noise
            observation[:,:,1] = observation[:,:,1]*_noise
            observation[:,:,2] = observation[:,:,2]*_noise
        return observation

    def _mask(self, observation):
        """对 obs 中某一片 movement info 遮住
        """
        if np.random.rand()>0.5:
            self.logger.debug(f'Add mask in movement info.')
            _slice = list(range(observation.shape[0]))
            _mask_index = np.random.choice(_slice)
            observation[_mask_index,:,[3,4]] = 0 # 将 mask_index 的置 0
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
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info


    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs = self.env.reset(**kwargs)

        return self.observation(obs)