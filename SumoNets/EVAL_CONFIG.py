'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 测试所使用的路网
@LastEditTime: 2023-02-24 23:20:10
'''
EVAL_SUMO_CONFIG = dict(
    # 四路口, 车道数 (3,3,3,3)
    eval_four_3=dict(
        tls_id = 'J1',
        sumocfg = 'train_four_3.sumocfg', # 对应的配置文件
        nets = ['4phases.net.xml'],
        routes = ['0.rou.xml'],
        start_time = 0, # route 开始时间
        edges = ['E0', '-E2', '-E1', '-E3'], # 存储 edge 的名称
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
            'NS-SN':['-E3 E2', '-E2 E3']
        }
    ),
)