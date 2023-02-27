'''
@Author: WANG Maonan
@Date: 2022-06-16 21:51:03
@Description: 对 obs 进行 shuffle，保证一个 frame stack 中的 obs shuffle 顺序是一样的
@LastEditTime: 2022-08-10 20:33:13
'''
import logging
import gym
import numpy as np

class shuffle_wrapper(gym.ObservationWrapper):
    """
    """
    def __init__(
        self,
        env: gym.Env,
        is_shuffle: str
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
        """
        super().__init__(env)
        self.logger = logging.getLogger(__name__)
        self.is_shuffle = is_shuffle # 是否 shuffle
        _phase_num = self.observation_space.shape[-2]
        assert _phase_num == 8, '相位数是 8.'
        self._idx = list(range(_phase_num))
        

    def observation(self, observation):
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
                    [7, 7, 7]]]
                )
        """
        if self.is_shuffle:
            np.random.shuffle(self._idx)
        _obs = observation[:]
        self.logger.debug(f'Shuffle, {self.is_shuffle}, --> {self._idx}')
        return _obs[:, self._idx] # 进行乱序

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