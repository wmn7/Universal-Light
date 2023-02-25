'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: SUMO Net 的配置文件, 记录每个文件夹包含的:
-> nets, routes, sumocfg
-> junction 的 ID
-> 路网的 edges
-> edges 的连接关系
这里需要注意，每个文件夹包含的路网只有相位结构不一样，其他都是一样的
@LastEditTime: 2023-02-24 23:20:10
'''
SUMO_NET_CONFIG = dict(
    # 四路口, 车道数 (3,3,3,3)
    train_four_3=dict(
        tls_id = 'J1',
        sumocfg = 'train_four_3.sumocfg', # 对应的配置文件
        nets = ['2phases.net.xml', '4phases.net.xml', '4phases_s.net.xml'],
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
    # 三路口, 车道数 (3,4,4)
    train_three_34=dict(
        tls_id = 'J1',
        sumocfg = 'train_three_34.sumocfg',
        nets = ['3phases.net.xml', '3phases_s.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0,
        edges = ['E0', '-E1', '-E3'],
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
        }
    ),
    # 四路口, 车道数 (3,3,3,3)
    test_four_3=dict(
        tls_id = 'J1',
        sumocfg = 'train_three_3.sumocfg',
        nets = ['6phases.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0,
        edges = ['E0', '-E2', '-E1', '-E3'],
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
            'NS-SN':['-E3 E2', '-E2 E3']
        }
    ),
    # 四路口, 车道数 (3,3,4,4)
    test_four_34=dict(
        tls_id = 'J1',
        sumocfg = 'test_four_34.sumocfg',
        nets = ['4phases.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0,
        edges = ['E0', '-E1', '-E3'],
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
            'NS-SN':['-E3 E2', '-E2 E3']
        }
    ),
    # 三路口, 车道数 (3,4,4)
    test_three_34=dict(
        tls_id = 'J1',
        sumocfg = 'test_three_34.sumocfg',
        nets = ['3phases.net.xml'],
        routes = ['0.rou.xml', '1.rou.xml', '2.rou.xml', '3.rou.xml', '4.rou.xml'],
        start_time = 0,
        edges = ['E0', '-E1', '-E3'],
        connections = {
            'WE-EW':['E0 E1', '-E1 -E0'],
        }
    ),
    # 四路口, 车道数 (2,2,2,2)
    cologne1=dict(
        tls_id = 'cluster_357187_359543',
        sumocfg = 'cologne1.sumocfg',
        nets = ['cologne1.net.xml'],
        routes = ['cologne1.rou.xml'],
        start_time = 25205,
        edges = ['-32038056#3', '23429231#1', '28198821#3', '27115123#3'],
        connections = {
            'WE-EW':['-32038056#3 -28198821#4', '28198821#3 32038056#0'],
            'NS-SN':['23429231#1 32038051#0', '27115123#3 32324544#0']
        }
    ),
    # 三路口, 车道数 (2,2,3)
    ingolstadt1=dict(
        tls_id = 'cluster_274083968_cluster_1200364014_1200364088',
        sumocfg = 'ingolstadt1.sumocfg',
        nets = ['ingolstadt1.net.xml'],
        routes = ['ingolstadt1.rou.xml'],
        start_time = 57600,
        edges = ['201963537#1', '164051413', '104010354'],
        connections = {
            'WE-EW':['201963537#1 104010475#0', '104010354 124812857#0'],
        }
    ),
)