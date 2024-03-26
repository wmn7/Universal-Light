'''
@Author: WANG Maonan
@Date: 2024-03-24 15:40:51
@Description: 分析 Route 的变化趋势
@LastEditTime: 2024-03-24 16:08:04
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.analysis_output.route_analysis import count_vehicles_for_multiple_edges, plot_vehicle_counts

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

route_file = current_file_path('./4.rou.xml')

edge_vehs = count_vehicles_for_multiple_edges(
    xml_path=route_file,
    edges_list=['E0 E1', '-E1 -E0', '-E3 E2', '-E2 E3'],
    interval=300
)
plot_vehicle_counts(edge_vehs, current_file_path('./route.png'))