'''
@Author: WANG Maonan
@Date: 2022-10-08 12:26:45
@Description: 分析每秒进入车辆, 并绘制图像
@LastEditTime: 2023-02-24 23:21:20
'''
import scienceplots
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science','ieee','no-latex'])

from aiolos.utils.get_abs_path import getAbsPath
from aiolos.outputAnalyze import RouteAnalysis

from net_config import SUMO_NET_CONFIG


def moving_average(x, w):
    """使得车流变得平滑

    Args:
        x (_type_): 每分钟车流的数据
        w (_type_): 平滑窗口

    Returns:
        _type_: _description_
    """
    return np.convolve(x, np.ones(w), 'valid') / w
    
if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)

    FOLDER_NAME = 'train_four_3' # 不同类型的路口
    route_name = SUMO_NET_CONFIG[FOLDER_NAME]['routes'][0] # 要分析的路网
    start_time = SUMO_NET_CONFIG[FOLDER_NAME]['start_time'] # route 开始的时间
    route_xml = pathConvert(f'./{FOLDER_NAME}/routes/{route_name}') # 读取 route.rou.xml 文件
    r_analysis = RouteAnalysis(output_file=route_xml)

    # 获得链接
    net_connections = SUMO_NET_CONFIG[FOLDER_NAME]['connections']
    directions_edges = dict()
    for _direction, _connections in net_connections.items():
        directions_edges[_direction] = _connections
    directions_vehicles = dict()

    time_intervals = 60 # 统计时间的间隔
    for _direction, _edges in directions_edges.items():
        vehicles_per_second = r_analysis.parse_data(directions=_edges, intervals=time_intervals, start_time=start_time) # 统计传入的所有 edge 的流量
        vehicles_per_second = sorted(vehicles_per_second.items(), key=lambda d:d[0]) # 按照 key (时间) 进行排序
        directions_vehicles[_direction] = [i[1]/time_intervals for i in vehicles_per_second]

    # 绘制图像
    fig1, ax1 = plt.subplots(figsize=plt.figaspect(0.6))
    for edge_id, edge_flow in directions_vehicles.items():
        smooth_flow = moving_average(edge_flow, 1)
        ax1.plot(
            list(range(0, len(smooth_flow))), 
            np.array(smooth_flow), 
            label=f'{edge_id}'
        )
    ax1.set_ylabel("Vehicles / Seconds")
    ax1.set_xlabel('Time (minutes)')
    # ax1.set(ylim=[0, 0.7]) # 限制 x 和 y 轴的范围
    plt.legend()
    plt.savefig(pathConvert(f'./{FOLDER_NAME}/vis_flow.png'))