'''
@Author: WANG Maonan
@Date: 2023-11-01 23:39:42
@Description: 绘制 Reward Curve with Standard Deviation
@LastEditTime: 2024-03-26 22:46:33
'''
from tshub.utils.plot_reward_curves import plot_reward_curve
from tshub.utils.get_abs_path import get_abs_path
path_convert = get_abs_path(__file__)


if __name__ == '__main__':
    log_files = [
        path_convert(f'./log/{i}.monitor.csv')
        for i in range(10)
    ]
    output_file = path_convert('./reward.png')
    plot_reward_curve(log_files, output_file, window_size=3)
