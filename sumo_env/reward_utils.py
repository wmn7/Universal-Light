'''
@Author: WANG Maonan
@Date: 2024-03-23 19:15:53
@Description: 奖励归一化 & 固定范围标准化
@LastEditTime: 2024-03-23 21:23:28
'''
class FixedRangeNormalizer:
    def __init__(self):
        self.env_extremes = {
            "train_four_3__2phases.net.xml": {'min': 0.09, 'max': 1},
            "train_four_3__4phases.net.xml": {'min': 0.043, 'max': 1},
            "train_four_3__4phases_s.net.xml": {'min': 0.043, 'max': 1},
            "train_four_345__4phases.net.xml": {'min': 0.043, 'max': 1},
            "train_four_345__4phases_s.net.xml": {'min': 0.042, 'max': 1},
            "train_four_345__6phases.net.xml": {'min': 0.037, 'max': 1},
            "train_three_3__3phases.net.xml": {'min': 0.05, 'max': 1},
            "train_three_3__3phases_s.net.xml": {'min': 0.05, 'max': 1},
        }  # Dictionary to store min and max rewards for each environment

    def update(self, key, reward) -> None:
        # If the environment is new, initialize its min and max with the first reward
        if key not in self.env_extremes:
            self.env_extremes[key] = {'min': reward, 'max': reward}
        else:
            # Update the min and max values for the environment if the new reward is outside the current range
            self.env_extremes[key]['min'] = min(self.env_extremes[key]['min'], reward)
            self.env_extremes[key]['max'] = max(self.env_extremes[key]['max'], reward)

    def normalize(self, key, reward):
        # Retrieve the min and max rewards for the environment
        min_reward = self.env_extremes[key]['min']
        max_reward = self.env_extremes[key]['max']
        
        # Avoid division by zero if min and max are the same
        if min_reward == max_reward:
            return 0
        
        # Normalize the reward to the range around [0, 1]
        normalized_reward = (reward - min_reward) / (max_reward - min_reward)
        return normalized_reward


class RewardNormalizer:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0
        self.mean_diff = 0
        self.var = 0

    def update(self, reward) -> None:
        self.n += 1
        last_mean = self.mean
        self.mean += (reward - last_mean) / self.n
        self.mean_diff += (reward - last_mean) * (reward - self.mean)
        self.var = self.mean_diff / self.n if self.n > 1 else 0

    def normalize(self, reward):
        if self.n < 2:
            return reward
        std = self.var ** 0.5
        return (reward - self.mean) / (std + 1e-8)  # Add a small constant to prevent division by zero