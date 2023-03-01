'''
@Author: WANG Maonan
@Date: 2023-03-01 16:47:09
@Description: 测试 FixTime 的结果
1. 输入每个相位的持续时间，进行仿真；
2. 得到 SUMO 仿真的结果数据
@LastEditTime: 2023-03-01 17:33:33
'''
import os
import gym
from typing import List, Callable, Dict

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.trafficLog.initLog import init_logging
from aiolos.AssembleEnvs.SetPhaseDurationDiscreteEnv import SetPhaseDurationDiscreteSUMOEnvironment

from .fix_time_wrapper import env_wrapper

def make_env(            
            tls_id:str,
            begin_time:int,
            num_seconds:int,
            sumo_cfg:str,
            net_file:str,
            route_file:str,
            is_libsumo:bool=False,
            trip_info:str=None,
            statistic_output:str=None,
            tls_state_add:List[str]=None,
            summary:str=None,
            queue_output:str=None,
            min_green:int=5,
            yellow_times:int=3,
            use_gui:bool=False,
            mode='train',
        ) -> Callable:
    """
    创建 Set Phase Duration Discrete, 离散的修改 phase duration

    Args:
        tls_id (str): 控制的信号灯的 id
        num_seconds (int): 总的仿真时间, 到达仿真时间则结束
        sumo_cfg (str): sumo config 文件
        net_files str: net 文件
        route_files str: route 文件
        trip_info (str, optional): 如果不是 None, 则输出仿真过程的 trip_info, 包含每辆车的信息; 如果是 None, 则不输出. Defaults to None.
        statistic_output (str, optional): 如果不是 None, 则输出仿真过程的 statistic out, 包含仿真总的等待时间等; 如果是 None, 则不输出.. Defaults to None.
        tls_state_add (List, optional): 添加指定的 tls add 文件, 输出信号灯的变化情况. Defaults to None.
        min_green (int, optional): 最小绿灯时间. Defaults to 5.
        use_gui (bool, optional): 是否使用 sumo-gui 打开. Defaults to False.
    """
    def _init() -> gym.Env:
        pathConvert = getAbsPath(__file__)
        init_logging(log_path=pathConvert('../'), prefix=f'PID_{os.getpid()}', log_level=0)

        env = SetPhaseDurationDiscreteSUMOEnvironment(
                            sumo_cfg=sumo_cfg,
                            begin_time=begin_time,
                            net_file=net_file,
                            route_file=route_file,
                            trip_info=trip_info,
                            statistic_output=statistic_output,
                            tls_state_add=tls_state_add,
                            summary=summary,
                            queue_output=queue_output,
                            use_gui=use_gui,
                            is_libsumo=is_libsumo,
                            tls_list=[tls_id],
                            num_seconds=num_seconds,
                            min_greens={tls_id:min_green},
                            yellow_times={tls_id:yellow_times},
                            delta_times={tls_id:None},
                            is_movement=True
                        ) # 创建 Set Current Phase Durations 环境
        
        env = env_wrapper(
            env=env,
            tls_id=tls_id,
            mode=mode, # 测试结束之后不需要进行 reset
        )
        return env
        
    return _init