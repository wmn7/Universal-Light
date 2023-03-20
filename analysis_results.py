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
    """分析 statistic.out.xml 中的 waiting time
    """
    tree = ET.parse(file_path)

    for children in tree.iter():
        if children.tag=="vehicleTripStatistics":
            waiting_time = children.attrib["waitingTime"]
            return float(waiting_time)


if __name__ == '__main__':
    pathConvert = getAbsPath(__file__)

    net_folders = [
        'train_four_3', 'train_four_345', 'train_three_3', 
        'test_four_34', 'test_three_34', 'ingolstadt1'
    ] # 所有测试的路网, cologne1

    for _net_folder in net_folders:
        nets_name = SUMO_CONFIG[_net_folder]['nets'] # network file (包含多个路网文件)
        routes_name = SUMO_CONFIG[_net_folder]['routes'] # route file (包含多个车辆文件)

        for _net_name in nets_name:
            _net_name = _net_name.split('.')[0] # 获得路网名称
            # ############
            # exp1 & exp2
            # ############
            # scnn 类模型
            scnn_results = list()
            scnn_all_results = list()
            # ernn 类模型
            ernn_results = list()
            ernn_all_results = list()
            # eattention 类模型
            eattention_results = list()
            eattention_all_results = list()

            # ####################
            # exp3, 数据增强结果分析
            # ####################
            shuffle_results = list()
            shuffle_changeLanes_results = list()
            shuffle_flowScale_results = list()
            shuffle_noise_results = list()
            shuffle_mask_results = list()

            changeLanes_results = list()
            changeLanes_flowScale_results = list()
            changeLanes_noise_results = list()
            changeLanes_mask_results = list()

            flowScale_results = list()
            flowScale_noise_results = list()
            flowScale_mask_results = list()

            noise_results = list()
            noise_mask_results = list()

            mask_results = list()

            for _route_name in routes_name:
                _route_name = _route_name.split('.')[0]
                # #######
                # 文件路径
                # #######
                # exp1 & exp2
                scnn_path = pathConvert(f'./results/output/scnn/6_0_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                scnn_all_path = pathConvert(f'./results/output/scnn/6_0_True_True_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                ernn_path = pathConvert(f'./results/output/ernn/6_0_False_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                ernn_all_path = pathConvert(f'./results/output/ernn/6_0_True_True_True_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                eattention_path = pathConvert(f'./results/output/eattention/6_0_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                eattention_all_path = pathConvert(f'./results/output/eattention/6_0_True_True_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')

                # exp3, 顺序是 {N_STACK}_{N_DELAY}_{SHFFLE}_{CHANGE_LANE}_{FLOW_SCALE}_{MASK}_{NOISE}
                shuffle_path = pathConvert(f'./results/exp3/output/6_0_True_False_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                shuffle_changeLanes_path = pathConvert(f'./results/exp3/output/6_0_True_True_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                shuffle_flowScale_path = pathConvert(f'./results/exp3/output/6_0_True_False_True_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                shuffle_noise_path = pathConvert(f'./results/exp3/output/6_0_True_False_False_True_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                shuffle_mask_path = pathConvert(f'./results/exp3/output/6_0_True_False_False_False_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')

                changeLanes_path = pathConvert(f'./results/exp3/output/6_0_False_True_False_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                changeLanes_flowScale_path = pathConvert(f'./results/exp3/output/6_0_False_True_True_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                changeLanes_noise_path = pathConvert(f'./results/exp3/output/6_0_False_True_False_True_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                changeLanes_mask_path = pathConvert(f'./results/exp3/output/6_0_False_True_False_False_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')

                flowScale_path = pathConvert(f'./results/exp3/output/6_0_False_False_True_False_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                flowScale_noise_path = pathConvert(f'./results/exp3/output/6_0_False_False_True_True_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                flowScale_mask_path = pathConvert(f'./results/exp3/output/6_0_False_False_True_False_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')

                noise_path = pathConvert(f'./results/exp3/output/6_0_False_False_False_True_False/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')
                noise_mask_path = pathConvert(f'./results/exp3/output/6_0_False_False_False_True_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')

                mask_path = pathConvert(f'./results/exp3/output/6_0_False_False_False_False_True/{_net_folder}/{_net_name}/{_route_name}/statistic.out.xml')                

                # ##########
                # 分析文件结果
                # ##########
                # exp1 & exp2
                scnn_wt = get_statistic_result(scnn_path)
                scnn_all_wt = get_statistic_result(scnn_all_path)
                ernn_wt = get_statistic_result(ernn_path)
                ernn_all_wt = get_statistic_result(ernn_all_path)
                eattention_wt = get_statistic_result(eattention_path)
                eattention_all_wt = get_statistic_result(eattention_all_path)

                # exp3
                shuffle_wt = get_statistic_result(shuffle_path)
                shuffle_changeLanes_wt = get_statistic_result(shuffle_changeLanes_path)
                shuffle_flowScale_wt = get_statistic_result(shuffle_flowScale_path)
                shuffle_noise_wt = get_statistic_result(shuffle_noise_path)
                shuffle_mask_wt = get_statistic_result(shuffle_mask_path)

                changeLanes_wt = get_statistic_result(changeLanes_path)
                changeLanes_flowScale_wt = get_statistic_result(changeLanes_flowScale_path)
                changeLanes_noise_wt = get_statistic_result(changeLanes_noise_path)
                changeLanes_mask_wt = get_statistic_result(changeLanes_mask_path)

                flowScale_wt = get_statistic_result(flowScale_path)
                flowScale_noise_wt = get_statistic_result(flowScale_noise_path)
                flowScale_mask_wt = get_statistic_result(flowScale_mask_path)

                noise_wt = get_statistic_result(noise_path)
                noise_mask_wt = get_statistic_result(noise_mask_path)

                mask_wt = get_statistic_result(mask_path)

                # ###########
                # 添加文件结果
                # ###########
                # exp1 & exp2
                scnn_results.append(scnn_wt)
                scnn_all_results.append(scnn_all_wt)
                ernn_results.append(ernn_wt)
                ernn_all_results.append(ernn_all_wt)
                eattention_results.append(eattention_wt)
                eattention_all_results.append(eattention_all_wt)

                # exp3
                shuffle_results.append(shuffle_wt)
                shuffle_changeLanes_results.append(shuffle_changeLanes_wt)
                shuffle_flowScale_results.append(shuffle_flowScale_wt)
                shuffle_noise_results.append(shuffle_noise_wt)
                shuffle_mask_results.append(shuffle_mask_wt)

                changeLanes_results.append(changeLanes_wt)
                changeLanes_flowScale_results.append(changeLanes_flowScale_wt)
                changeLanes_noise_results.append(changeLanes_noise_wt)
                changeLanes_mask_results.append(changeLanes_mask_wt)

                flowScale_results.append(flowScale_wt)
                flowScale_noise_results.append(flowScale_noise_wt)
                flowScale_mask_results.append(flowScale_mask_wt)

                noise_results.append(noise_wt)
                noise_mask_results.append(noise_mask_wt)

                mask_results.append(mask_wt)        

            print(
                f'{_net_folder}-{_net_name}:\n'
                f'--- Exp1 & Exp2 ---\n'
                f'SCNN: {scnn_results};\n'
                f'SCNN+ALL: {scnn_all_results};\n'
                f'ERNN: {ernn_results};\n'
                f'ERNN+ALL: {ernn_all_results};\n'
                f'EAttention: {eattention_results};\n'
                f'EAttention+ALL: {eattention_all_results};\n'
                f'--- Exp3 ---\n'
                f'Shuffle, {shuffle_results};\n'
                f'Shuffle + ChangeLanes, {shuffle_changeLanes_results};\n'
                f'Shuffle + FlowScale, {shuffle_flowScale_results};\n'
                f'Shuffle + Noise, {shuffle_noise_results};\n'
                f'Shuffle + Mask, {shuffle_mask_results};\n'
                f'ChangeLanes, {changeLanes_results};\n'
                f'ChangeLanes + FlowScale, {changeLanes_flowScale_results};\n'
                f'ChangeLanes + Noise, {changeLanes_noise_results};\n'
                f'ChangeLanes + Mask, {changeLanes_mask_results};\n'
                f'FlowScale, {flowScale_results};\n'
                f'FlowScale + Noise, {flowScale_noise_results};\n'
                f'FlowScale + Mask, {flowScale_mask_results};\n'
                f'Noise, {noise_results};\n'
                f'Noise + Mask, {noise_mask_results};\n'
                f'Mask, {mask_results};\n'
            )