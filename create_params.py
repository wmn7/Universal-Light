'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 创建训练和测试环境的参数
@LastEditTime: 2023-02-24 23:20:10
'''
from aiolos.utils.get_abs_path import getAbsPath

from SumoNets.TRAIN_CONFIG import TRAIN_SUMO_CONFIG
from SumoNets.EVAL_CONFIG import EVAL_SUMO_CONFIG

def create_params(
        is_eval:bool, is_shuffle:bool, is_change_lane:bool, is_noise:bool, is_mask:bool,
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

    if is_eval:
        env_dict = {
            _folder: {
                'cfg': pathConvert(f'./SumoNets/{_folder}/env/{EVAL_SUMO_CONFIG[_folder]["sumocfg"]}'),
                'net':[pathConvert(f'./SumoNets/{_folder}/env/{_net}') for _net in EVAL_SUMO_CONFIG[_folder]['nets']],
                'route':[pathConvert(f'./SumoNets/{_folder}/routes/{_route}') for _route in EVAL_SUMO_CONFIG[_folder]['routes']]
            }
            for _folder in ['train_four_3']
        }
    else:
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