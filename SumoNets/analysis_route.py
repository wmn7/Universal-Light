'''
@Author: WANG Maonan
@Date: 2023-02-17 21:12:26
@Description: 分析 route 文件, 得到车辆数, 车流到达的平均值
@LastEditTime: 2023-02-24 23:21:10
'''
import numpy as np

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.outputAnalyze import RouteAnalysis

from net_config import SUMO_NET_CONFIG

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
    
if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)
    FOLDER_NAME = 'train_four_3' # 不同类型的路口

    # 获得 route 文件
    ROUTES = SUMO_NET_CONFIG[FOLDER_NAME]['routes'] # 路网的名称
    for ROUTE in ROUTES:
        route_xml = pathConvert(f'./{FOLDER_NAME}/routes/{ROUTE}') # 读取 route.rou.xml 文件
        
        # 获得对应的 connection
        net_connections = SUMO_NET_CONFIG[FOLDER_NAME]['connections']
        EDGES = list()
        for _, _connections in net_connections.items():
            EDGES += _connections

        r_analysis = RouteAnalysis(output_file=route_xml)

        time_intervals = 60 # 统计时间的间隔
        vehicles_per_minutes = r_analysis.parse_data(directions=EDGES, intervals=time_intervals) # 统计传入的所有 edge 的流量
        vehicles_per_minutes = sorted(vehicles_per_minutes.items(), key=lambda d:d[0]) # 按照 key (时间) 进行排序
        vehicles_per_second = [i[1]/time_intervals for i in vehicles_per_minutes] # 转换为每秒多少车
        print(
            f'Total Vehicle Number, {np.sum([i[1] for i in vehicles_per_minutes])}\n',
            f'Mean, {np.mean(vehicles_per_second)}\n',
            f'Std, {np.std(vehicles_per_second)}\n',
            f'Max, {np.max(vehicles_per_second)}\n',
            f'Min, {np.min(vehicles_per_second)}',
        )