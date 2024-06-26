'''
@Author: WANG Maonan
@Date: 2022-03-15 12:02:58
@Description: 测试模型 eattention_cls 的输出, 和 eattention 比起来, 加入了 cls token, 简化了最终的参数
@LastEditTime: 2024-03-26 22:08:31
'''
import sys
from tshub.utils.get_abs_path import get_abs_path
pathConvert = get_abs_path(__file__)
sys.path.append(pathConvert('../../'))

import torch
import numpy as np
import gymnasium as gym

from model_structures.eattention_cls import EAttentionCLS

if __name__ == '__main__':
    # Input 是一个 N*8*8 的矩阵
    observation_space = gym.spaces.Box(
            low=0, 
            high=5,
            shape=(4,8,8)
        ) # obs 空间
    net = EAttentionCLS(observation_space, features_dim=32)

    movement_info = np.array([
        0.3*np.ones((4,8,8)), 
        0.5*np.ones((4,8,8)),
        0.7*np.ones((4,8,8)),
    ]) # (3, 4, 8, 8)

    movement_info = torch.from_numpy(movement_info).to(torch.float32) # batch_size*movement_num*movement_info_dim
    result = net(movement_info)
    print(result.shape) # 提取特征
