'''
@Author: WANG Maonan
@Date: 2022-03-15 12:02:58
@Description: 创建 Next or Not 环境
整个环境创建的流程如下所示：
- 使用 SetPhaseDurationDiscreteSUMOEnvironment 创建动作为 Next or Not 的环境
- Env Wrapper 的使用
1. 利用 obs_reward_wrapper 处理 obs, action 和 reward
2. 将 obs 组成队列，实现 frame stack 和 delayed obs
3. 对观测值进行增强

下面是参数解释：

1. is_shuffle, 是否使用数据增强
2. num_stack, 是否将 N 个时间段的数据进行拼接
3. num_delayed, 延迟 K 个时间
@LastEditTime: 2022-06-21 22:05:00
'''
import os
import gym
from typing import List, Callable, Dict
from stable_baselines3.common.monitor import Monitor
from aiolos.AssembleEnvs.KeepChangePhaseEnv import KeepChangePhaseSUMOEnvironment

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging

from .obs_reward_wrapper import obs_reward_wrapper
from .frame_stack_wrapper import frame_stack_delayed_wrapper
from .data_augmentation_wrapper import data_augmentation_wrapper

def make_env(            
            tls_id:str,
            begin_time:int,
            num_seconds:int,
            sumo_cfg:str,
            net_files:List[str],
            route_files:List[str],
            log_file:str,
            env_index:str,
            env_dict:Dict[str,str]=None, # 这个优先级高于 net_file 和 route_files
            is_libsumo:bool=False,
            is_shuffle:bool=False, # 是否进行数据增强
            is_change_lane:bool=False,
            is_noise:bool=False,
            is_mask:bool=False,
            num_stack:int=4, # obs 堆叠的数量
            num_delayed:int=0, # obs 延迟的时间
            trip_info:str=None,
            statistic_output:str=None,
            summary:str=None,
            queue_output:str=None,
            tls_state_add:List[str]=None,
            delta_times:int=5,
            min_green:int=5,
            yellow_times:int=3,
            use_gui:bool=False,
            mode='train',
        ) -> Callable:
    """
    创建 Next or Not 的环境

    Args:
        tls_id (str): 控制的信号灯的 id
        begin_time (int): 仿真开始的时间
        num_seconds (int): 总的仿真时间, 到达仿真时间则结束
        sumo_cfg (str): sumo config 文件
        net_file (str): sumo net 文件
        log_file (str): 环境日志文件保存的文件夹
        route_files (List[str]): route 文件列表, reset 的时候会从中随机选择一个
        trip_info (str, optional): 如果不是 None, 则输出仿真过程的 trip_info, 包含每辆车的信息; 如果是 None, 则不输出. Defaults to None.
        statistic_output (str, optional): 如果不是 None, 则输出仿真过程的 statistic out, 包含仿真总的等待时间等; 如果是 None, 则不输出.. Defaults to None.
        tls_state_add (List, optional): 添加指定的 tls add 文件, 输出信号灯的变化情况. Defaults to None.
        min_green (int, optional): 最小绿灯时间. Defaults to 5.
        use_gui (bool, optional): 是否使用 sumo-gui 打开. Defaults to False.
    """
    def _init() -> gym.Env:
        pathConvert = getAbsPath(__file__)
        init_logging(log_path=pathConvert('../'), prefix=f'PID_{os.getpid()}', log_level=0)

        env = KeepChangePhaseSUMOEnvironment(
                            sumo_cfg=sumo_cfg,
                            begin_time=begin_time,
                            net_file=net_files[0],
                            route_file=route_files[0],
                            trip_info=trip_info,
                            statistic_output=statistic_output,
                            summary=summary,
                            queue_output=queue_output,
                            tls_state_add=tls_state_add,
                            use_gui=use_gui,
                            is_libsumo=is_libsumo,
                            tls_list=[tls_id],
                            num_seconds=num_seconds,
                            delta_times={tls_id:delta_times},
                            min_greens={tls_id:min_green},
                            yellow_times={tls_id:yellow_times},
                            is_movement=True
                        ) # 创建 Set Current Phase Duration 的环境
        
        env = obs_reward_wrapper(
                env=env,
                tls_id=tls_id,
                env_dict=env_dict, # 设置不同的路网
                is_movement=True,
                mode=mode, # 测试结束之后不需要进行 reset
        ) # 处理 obs, reward 和 action

        env = frame_stack_delayed_wrapper(
                env = env, 
                num_stack=num_stack,
                num_delayed=num_delayed
        ) # 处理 frame stack 和 delayed obs

        # data augmentation
        env = data_augmentation_wrapper(
            env=env, 
            is_shuffle=is_shuffle,
            is_change_lane=is_change_lane,
            is_noise=is_noise,
            is_mask=is_mask,
        ) # 对 obs 进行数据增强

        return Monitor(env, filename=f'{log_file}/{env_index}')
        
    return _init