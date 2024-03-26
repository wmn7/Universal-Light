'''
@Author: WANG Maonan
@Date: 2022-03-15 12:02:58
@Description: 测试模型 eattention 的输出
@LastEditTime: 2024-03-23 01:12:14
'''
import sys
from tshub.utils.get_abs_path import get_abs_path
pathConvert = get_abs_path(__file__)
sys.path.append(pathConvert('../../'))

import torch
import numpy as np
import gymnasium as gym

from models.eattention import EAttention


if __name__ == '__main__':
    # Input 是一个 N*8*8 的矩阵
    observation_space = gym.spaces.Box(
            low=0, 
            high=3,
            shape=(5,12,7)
        ) # obs 空间
    net = EAttention(observation_space, features_dim=32)

    movement_info = np.array([
        0.3*np.ones((5,12,7)), 
        0.5*np.ones((5,12,7)),
        0.7*np.ones((5,12,7)),
    ]) # (3, 4, 8, 8)

    movement_info = torch.from_numpy(movement_info).to(torch.float32) # batch_size*movement_num*movement_info_dim
    result = net(movement_info)
    print(result.shape) # 提取特征
