'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 对环境进行归一化
@LastEditTime: 2023-02-24 23:20:10
'''
import os
from stable_baselines3.common.callbacks import BaseCallback

class VecNormalizeCallback(BaseCallback):
    """保存环境标准化之后的值
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super(VecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {path}")
        return True


class VecBestNormalizeCallback(BaseCallback):
    """保存最优模型的环境值
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super(VecBestNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"best_{self.name_prefix}.pkl")
            self.model.get_vec_normalize_env().save(path)
            if self.verbose > 1:
                print(f"Saving VecNormalize to {path}")
        return True