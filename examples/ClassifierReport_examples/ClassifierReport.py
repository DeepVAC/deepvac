import sys
sys.path.append('../../lib')
from config import config as deepvac_config
import os
import numpy as np
import torch
import time
from syszux_report import ClassifierReport
from syszux_log import LOG


# get the closest and the second closest features
# input: 
#       min_distance: storage the result
#       db: db features
#       emb: current test feature
#       names: idx infomation
#
# output:
#       min_distance: storage the result
def getMinTup(min_distance, db, emb, names):
    distance = torch.norm(db - emb, dim=1)
    min_index = torch.argmin(distance).item()
    min_distance.append((names[min_index], distance[min_index].item()))
    sec_min_val, sec_min_index = distance.kthvalue(2)
    min_distance.append((names[sec_min_index], sec_min_val.item()))

    return min_distance

# generate report
# input:
#       dbs: db features list
#       names: idx info list
#       paths: img path list
#       file_path: result file
#       cls_num: number of ids
# output:
#       report: report info, use `report()` to call
def compareAndReport(dbs, names, paths, file_path, cls_num):

    done_paths = []
    done_infos = []
    if os.path.exists(file_path):
        with open(file_path, 'r')as f:
            lines = f.readlines()
        for line in lines:
            done_paths.append(line.split(' ')[0])
            done_infos.append(line)

    report = ClassifierReport('gemfield',cls_num, cls_num)
    f = open(file_path, 'w')
    if not len(done_infos) == 0:
        for inf in done_infos:
            _, label, _, pred = inf.strip().split(' ')
            report.add(int(label), int(pred))
            f.write(inf)

    total = 0

    for i, cur_db in enumerate(dbs):
        name = names[i]
        path = paths[i]
        for idx, emb in enumerate(cur_db):
            if path[idx] in done_paths:
                continue

            min_distance = []

            for n in range(len(dbs)):
                emb = emb.to(dbs[n].device)
                min_distance = getMinTup(min_distance, dbs[n], emb, names[n])
                
            sorted_min_distance = sorted(min_distance, key=lambda t:t[1])

            pred = sorted_min_distance[1][0]
            pred_ori = sorted_min_distance[0][0]
            #print('label : {}'.format(name[idx]))
            #print('pred : {}'.format(pred))
            LOG.logI("label : {}".format(name[idx]))
            LOG.logI("pred : {}".format(pred))
            if (total+idx) % 10000 == 0 and (total+idx) != 0:
                LOG.logI("Done {} img...".format(total+idx))
            report.add(int(name[idx]), int(pred))
            print("{} {} {} {}".format(path[idx], name[idx], pred_ori, pred))
            f.write("{} {} {} {}\n".format(path[idx], name[idx], pred_ori, pred))

        total += cur_db.shape
        LOG.logI("Total is {} now...".format(total))
    f.close()

    return report


def test(deepvac_config):
    db1 = torch.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_1.feature', map_location = {'cuda:0':'cuda:1'})
    db2 = torch.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_2.feature', map_location = {'cuda:0':'cuda:1'})
    db3 = torch.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_3.feature', map_location = {'cuda:1':'cuda:0'})
    db4 = torch.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_4.feature', map_location = {'cuda:1':'cuda:0'})
    np1 = np.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_1.feature.npz')
    np2 = np.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_2.feature.npz')
    np3 = np.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_3.feature.npz')
    np4 = np.load('/gemfield/hostpv/gemfield/deepvac-service/src/db/asia_emor_fix_merged_4.feature.npz')

    dbs = [db1, db2, db3, db4]
    names = [np1['names'], np2['names'], np3['names'], np4['names']]
    paths = [np1['paths'], np2['paths'], np3['paths'], np4['paths']]

    config_cls = deepvac_config.cls
    report = compareAndReport(dbs, names, paths, config_cls.file_path, config_cls.cls_num)
    report()


if __name__ == "__main__":
    #config_cls = deepvac_config.cls
    #report = compareAndReport(config_cls.feature_name, config_cls.file_path, config_cls.cls_num)
    #report()
    test(deepvac_config)

