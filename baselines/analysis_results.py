'''
@Author: WANG Maonan
@Date: 2022-06-21 22:33:40
@Description: 分析 sumo output 中的结果
@LastEditTime: 2022-07-08 20:03:25
'''
from xml.etree import ElementTree as ET
from aiolos.utils.get_abs_path import getAbsPath
pathConvert = getAbsPath(__file__)
import sys
sys.path.append(pathConvert('../'))

from SumoNets.NET_CONFIG import SUMO_NET_CONFIG as SUMO_CONFIG

def get_statistic_result(file_path):
    tree = ET.parse(file_path)

    for children in tree.iter():
        if children.tag=="vehicleTripStatistics":
            waiting_time = children.attrib["waitingTime"]
            return float(waiting_time)

if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)

    net_folders = [
        'train_four_3', 'train_four_345', 'train_three_3', 
        'test_four_34', 'test_three_34', 'cologne1', 'ingolstadt1'
    ] # 所有测试的路网

    for _net_folder in net_folders:
        nets_name = SUMO_CONFIG[_net_folder]['nets'] # network file (包含多个路网文件)
        routes_name = SUMO_CONFIG[_net_folder]['routes'] # route file (包含多个车辆文件)

        for _net_name in nets_name:
            _net_name = _net_name.split('.')[0]
            fix20_results = list()
            fix30_results = list()
            fix40_results = list()
            sotl_results = list()
            for _route_name in routes_name:
                _route_name = _route_name.split('.')[0]
                fix20_statistic_path = pathConvert(f'./Result/FixTime/{_net_folder}/20/{_net_name}/{_route_name}/statistic.out.xml')
                fix30_statistic_path = pathConvert(f'./Result/FixTime/{_net_folder}/30/{_net_name}/{_route_name}/statistic.out.xml')
                fix40_statistic_path = pathConvert(f'./Result/FixTime/{_net_folder}/40/{_net_name}/{_route_name}/statistic.out.xml')
                sotl_statistic_path = pathConvert(f'./Result/SOTL/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                fix20_wt = get_statistic_result(fix20_statistic_path)
                fix30_wt = get_statistic_result(fix30_statistic_path) # achieve the waiting time using FixTime30
                fix40_wt = get_statistic_result(fix40_statistic_path)
                sotl_wt = get_statistic_result(sotl_statistic_path) # 使用 sotl 的等待时间
                fix20_results.append(fix20_wt)
                fix30_results.append(fix30_wt)
                fix40_results.append(fix40_wt)
                sotl_results.append(sotl_wt)
            print(
                f'{_net_folder}-{_net_name}:\n'
                f'Fix20: {fix20_results};\n'
                f'Fix30: {fix30_results};\n'
                f'Fix40: {fix40_results};\n'
                f'SOTL: {sotl_results};\n'
                f'---\n'
            )