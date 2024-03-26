'''
@Author: WANG Maonan
@Date: 2023-02-15 14:33:49
@Description: 测试使用的路网
@LastEditTime: 2024-03-24 18:52:35
'''
TEST_SUMO_CONFIG = dict(
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
)