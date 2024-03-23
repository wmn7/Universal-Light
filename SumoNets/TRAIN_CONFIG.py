'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 训练使用的路网
@LastEditTime: 2024-03-24 01:32:31
'''
TRAIN_SUMO_CONFIG = dict(
    # 四路口, 车道数 (3,3,3,3)
    train_four_3=dict(
        tls_id = 'J1',
        sumocfg = 'train_four_3.sumocfg', # 对应的配置文件
        # nets = ['2phases.net.xml', '4phases.net.xml', '4phases_s.net.xml'],
        nets = ['4phases.net.xml', '4phases_s.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0, # route 开始时间
        edges = ['E0', '-E2', '-E1', '-E3'], # 存储 edge 的名称
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
            'NS-SN':['-E3 E2', '-E2 E3']
        }
    ),
    # 四路口, 车道数 (3,4,5,5)
    train_four_345=dict(
        tls_id = 'J1',
        sumocfg = 'train_four_345.sumocfg',
        nets = ['4phases.net.xml', '4phases_s.net.xml', '6phases.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0,
        edges = ['E0', '-E2', '-E1', '-E3'],
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
            'NS-SN':['-E3 E2', '-E2 E3']
        }
    ),
    # 三路口, 车道数 (3,3,3)
    train_three_3=dict(
        tls_id = 'J1',
        sumocfg = 'train_three_3.sumocfg',
        nets = ['3phases.net.xml', '3phases_s.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0,
        edges = ['E0', '-E1', '-E3'],
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
        }
    ),
)