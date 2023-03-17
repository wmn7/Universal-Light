'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 创建训练和测试环境的参数，这里有三个创建参数的函数：
- create_params，创建训练使用的参数
- create_test_params，创建测试使用的参数
- create_singleEnv_params，创建单个环境的参数
@LastEditTime: 2023-02-24 23:20:10
'''
import os
from aiolos.utils.get_abs_path import getAbsPath

from SumoNets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG
from SumoNets.EVAL_CONFIG import EVAL_SUMO_CONFIG
from SumoNets.NET_CONFIG import SUMO_NET_CONFIG # 测试模型时候训练和测试的路网一起进行测试
from SumoNets.TEST_CONFIG import TEST_SUMO_CONFIG

def create_params(
        is_eval:bool, 
        is_shuffle:bool, is_change_lane:bool, is_flow_scale:bool,
        is_noise:bool, is_mask:bool,
        N_STACK:int, N_DELAY:int, LOG_PATH:str
    ):
    pathConvert = getAbsPath(__file__)

    FOLDER_NAME = 'train_four_3' # 不同类型的路口
    tls_id = TRAIN_SUMO_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = TRAIN_SUMO_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    net_name = TRAIN_SUMO_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = TRAIN_SUMO_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = TRAIN_SUMO_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    # 转换为文件路径
    cfg_xml = pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{route_name}')]

    if is_eval: # 如果是测试的 reward，就只使用一个环境进行测试（使用 test 路网进行测试）
        env_dict = {
            _folder: {
                'cfg': pathConvert(f'./SumoNets/{_folder}/env/{EVAL_SUMO_CONFIG[_folder]["sumocfg"]}'),
                'net':[pathConvert(f'./SumoNets/{_folder}/env/{_net}') for _net in EVAL_SUMO_CONFIG[_folder]['nets']],
                'route':[pathConvert(f'./SumoNets/{_folder}/routes/{_route}') for _route in EVAL_SUMO_CONFIG[_folder]['routes']]
            }
            for _folder in ['test_four_34']
        }
    else: # 训练的时候多个路网同时进行训练
        env_dict = {
            _folder: {
                'cfg': pathConvert(f'./SumoNets/{_folder}/env/{TRAIN_SUMO_CONFIG[_folder]["sumocfg"]}'),
                'net':[pathConvert(f'./SumoNets/{_folder}/env/{_net}') for _net in TRAIN_SUMO_CONFIG[_folder]['nets']],
                'route':[pathConvert(f'./SumoNets/{_folder}/routes/{_route}') for _route in TRAIN_SUMO_CONFIG[_folder]['routes']]
            }
            for _folder in ['train_four_3', 'train_four_345', 'train_three_3']
        }

    params = {
        'tls_id':tls_id,
        'begin_time':start_time,
        'num_seconds':3600,
        'sumo_cfg':cfg_xml,
        'net_files':net_xml,
        'route_files':route_xml,
        'is_shuffle':is_shuffle,
        'is_change_lane':is_change_lane,
        'is_flow_scale':is_flow_scale,
        'is_noise':is_noise,
        'is_mask':is_mask,
        'num_stack':N_STACK,
        'num_delayed':N_DELAY,
        'is_libsumo':True,
        'use_gui':False,
        'min_green':5,
        'log_file':LOG_PATH,
        'env_dict':env_dict
    }

    return params



def create_test_params(
        net_name:str, output_folder:str,
        N_STACK:int, N_DELAY:int, LOG_PATH:str
    ):
    """对训练路网和测试路网都进行测试，这里的 mode=eval
    """
    pathConvert = getAbsPath(__file__)

    FOLDER_NAME = net_name # 要测试的路网名称
    tls_id = SUMO_NET_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = SUMO_NET_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    net_name = SUMO_NET_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = SUMO_NET_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = SUMO_NET_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    # 转换为文件路径
    cfg_xml = pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{route_name}')]

    route_params = dict() # 给每一个 route 生成一组参数
    for _net in SUMO_NET_CONFIG[FOLDER_NAME]['nets']:
        for _route in SUMO_NET_CONFIG[FOLDER_NAME]['routes']:
            _net_name = _net.split('.')[0] # 得到 NET 的名称
            _route_name = _route.split('.')[0] # 得到路网的名称
            route_output_folder = os.path.join(output_folder, f'{_net_name}/{_route_name}') # 模型输出文件夹
            os.makedirs(route_output_folder, exist_ok=True) # 创建文件夹
            trip_info = os.path.join(route_output_folder, f'tripinfo.out.xml')
            statistic_output = os.path.join(route_output_folder, f'statistic.out.xml')
            summary = os.path.join(route_output_folder, f'summary.out.xml')
            queue_output = os.path.join(route_output_folder, f'queue.out.xml')
            tls_add = [
                # 探测器
                pathConvert(f'./SumoNets/{FOLDER_NAME}/detectors/e1_internal.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/detectors/e2.add.xml'),
                # 信号灯
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_programs.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_state.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_switch_states.add.xml'),
                pathConvert(f'./SumoNets/{FOLDER_NAME}/add/tls_switches.add.xml')
            ]
        
            env_dict = {
                FOLDER_NAME: {
                    'cfg': pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{SUMO_NET_CONFIG[FOLDER_NAME]["sumocfg"]}'),
                    'net':[pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{_net}')],
                    'route':[pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{_route}')]
                }
            }

            params = {
                'tls_id':tls_id,
                'begin_time':start_time,
                'num_seconds':3600,
                'sumo_cfg':cfg_xml,
                'net_files':net_xml,
                'route_files':route_xml,
                'trip_info':trip_info,
                'statistic_output':statistic_output,
                'summary':summary,
                'queue_output':queue_output,
                'tls_state_add':tls_add,
                # 下面是数据增强的参数
                'is_shuffle':False,
                'is_change_lane':False,
                'is_flow_scale':False,
                'is_noise':False,
                'is_mask':False,
                'num_stack':N_STACK,
                'num_delayed':N_DELAY,
                # 下面是仿真器的参数
                'is_libsumo':True,
                'use_gui':False,
                'min_green':5,
                'log_file':LOG_PATH,
                'env_dict':env_dict,
                'mode':'eval'
            }

            _key = f'{_net_name}__{_route_name}'
            route_params[_key] = params

    return route_params



def create_singleEnv_params(
        net_name:str,
        is_shuffle:bool, is_change_lane:bool, is_flow_scale:bool,
        is_noise:bool, is_mask:bool,
        N_STACK:int, N_DELAY:int, 
        LOG_PATH:str
    ):
    """创建单个环境的参数, 输入 net name, 返回训练的参数
    """
    pathConvert = getAbsPath(__file__)

    FOLDER_NAME = net_name # 要测试的路网名称
    tls_id = TEST_SUMO_CONFIG[FOLDER_NAME]['tls_id'] # 路口 id
    cfg_name = TEST_SUMO_CONFIG[FOLDER_NAME]['sumocfg'] # sumo config
    net_name = TEST_SUMO_CONFIG[FOLDER_NAME]['nets'][0] # network file
    route_name = TEST_SUMO_CONFIG[FOLDER_NAME]['routes'][0] # route file
    start_time = TEST_SUMO_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间

    # 转换为文件路径
    cfg_xml = pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{cfg_name}')
    net_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')]
    route_xml = [pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{route_name}')]

    # 组成要测试的路网
    env_dict = {
        FOLDER_NAME: {
            'cfg': pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{TEST_SUMO_CONFIG[FOLDER_NAME]["sumocfg"]}'),
            'net':[pathConvert(f'./SumoNets/{FOLDER_NAME}/env/{net_name}')],
            'route':[pathConvert(f'./SumoNets/{FOLDER_NAME}/routes/{_route}') for _route in TEST_SUMO_CONFIG[FOLDER_NAME]['routes']]
        }
    }

    params = {
        'tls_id':tls_id,
        'begin_time':start_time,
        'num_seconds':3600,
        'sumo_cfg':cfg_xml,
        'net_files':net_xml,
        'route_files':route_xml,
        # 下面是数据增强的参数
        'is_shuffle':is_shuffle,
        'is_change_lane':is_change_lane,
        'is_flow_scale':is_flow_scale,
        'is_noise':is_noise,
        'is_mask':is_mask,
        'num_stack':N_STACK,
        'num_delayed':N_DELAY,
        # 下面是仿真器的参数
        'is_libsumo':True,
        'use_gui':False,
        'min_green':5,
        'log_file':LOG_PATH,
        'env_dict':env_dict,
    }

    return params