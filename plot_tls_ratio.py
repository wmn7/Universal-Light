'''
@Author: WANG Maonan
@Date: 2022-11-22 22:20:47
@Description: 绘制某个方向绿灯时间占比的变化，查看是否与流量是对应的，找测试例子即可
@LastEditTime: 2022-11-23 17:57:28
'''
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use(['science','ieee', 'grid', 'vibrant', 'no-latex'])
plt.style.use(['science', 'grid', 'vibrant', 'no-latex'])


from aiolos.utils.get_abs_path import getAbsPath
from aiolos.outputAnalyze import TlsProgramAnalysis
from SumoNets.NET_CONFIG import SUMO_NET_CONFIG

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)
    net_type = 'ingolstadt1'
    tls_id = SUMO_NET_CONFIG[net_type]['tls_id']
    net_id = SUMO_NET_CONFIG[net_type]['nets'][0].split('.')[0]
    route_id = SUMO_NET_CONFIG[net_type]['routes'][0].split('.')[0]
    tls_state_xml = pathConvert(f'./results/exp3/output/6_0_True_True_True_True_True/{net_type}/{net_id}/{route_id}/add/tls_program.out.xml') # 读取 route.rou.xml 文件
    tls_analysis = TlsProgramAnalysis(output_file=tls_state_xml)

    yellow_time = 12 # 一个周期内会有的黄灯时间
    states = SUMO_NET_CONFIG[net_type]['states']
    state_time_dict = dict() # 存储所有 state 的时间
    state_ratio_dict = {
        _state:list()
        for _state in states
    } # 存储 state 在一个周期内的时间占比
    cumulative_cycle_length = [0] # 计算每个周期的时间长度, 计算每个周期的开始和结束时间
    
    directions_state = SUMO_NET_CONFIG[net_type]['directions_state']

    # 得到具体的信号灯时间
    for _state in states:
        state_per_interval = tls_analysis.parse_data(tls_id=tls_id, state=_state)
        state_time_dict[_state] = state_per_interval # 存储所有 state 的时间

    # 转换为绿灯的占比
    data_amount = min([len(state_time_dict[_state]) for _state in states]) # 周期数量
    for i in range(data_amount):
        _cycle_length = sum([state_time_dict[_state][i] for _state in states]) # 计算周期长度
        cumulative_cycle_length.append((cumulative_cycle_length[-1] + _cycle_length + yellow_time)) # 保存周期的结束时间
        for _state in states:
            _ratio = state_time_dict[_state][i] / _cycle_length # 计算占比
            state_ratio_dict[_state].append(_ratio)
    

    # 绘制图像
    fig1, ax1 = plt.subplots(figsize=plt.figaspect(0.6))
    for edge_id, edge_state in directions_state.items():
        _time = [(i+j)/2/60 for i,j in zip(cumulative_cycle_length[1:], cumulative_cycle_length[:-1])]
        _ratio = np.array(state_ratio_dict[edge_state])
        smooth_ratio = moving_average(_ratio, 3)
        ax1.plot(
            _time[:len(smooth_ratio)], # 转换为秒
            smooth_ratio, # 对 ratio 进行滑动平均
            label=f'{edge_id}'
        )
    ax1.set_ylabel("Green Time Ratio")
    ax1.set_xlabel('Time (Minutes)')
    # ax1.set(ylim=[0, 0.7]) # 限制 x 和 y 轴的范围
    plt.legend(loc=2)
    plt.savefig(pathConvert(f'./tls_ratio_results.pdf'))