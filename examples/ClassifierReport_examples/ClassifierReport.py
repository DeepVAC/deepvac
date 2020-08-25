import sys
sys.path.append('../../lib')
from config import config as deepvac_config
import os
import numpy as np
import torch
import time
from syszux_report import ClassifierReport
from syszux_log import LOG


# get the nearest and the second nearest features from a db
# input: 
#       db: db features
#       emb: current test feature
#       names: idx infomation
#
# output:
#       min_distance: storage the result
def getNearestTwoFeaturesFromDB(db, emb, names):
    min_distance = []
    distance = torch.norm(db - emb, dim=1)
    min_index = torch.argmin(distance).item()
    min_distance.append((names[min_index], distance[min_index].item()))
    sec_min_val, sec_min_index = distance.kthvalue(2)
    min_distance.append((names[sec_min_index], sec_min_val.item()))

    return min_distance

# get the nearest and the second nearest features from all db
# input:
#       emb: a img feature for compare
#       dbs: db features list
#       names: idx info list
# output:
#       the nearest idx(name) and the second nearest idx(name)
def getNearestTwoFeatureFromAllDB(emb, dbs, names):
    min_distance = []
    for n in range(len(dbs)):
        emb = emb.to(dbs[n].device)
        min_distance.extend(getNearestTwoFeaturesFromDB(dbs[n], emb, names[n]))

    sorted_min_distance = sorted(min_distance, key=lambda t:t[1])

    return sorted_min_distance[0][0], sorted_min_distance[1][0]

# generate report
# input:
#       dbs: db features list
#       names: idx info list
#       paths: img path list
#       file_path: result file
#       cls_num: number of ids
# output:
#       report: report info, use `report()` to call
def getClassifierReport(dbs, names, paths, file_path, cls_num):

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

    for i, db in enumerate(dbs):
        name = names[i]
        path = paths[i]
        for idx, emb in enumerate(db):
            if path[idx] in done_paths:
                continue

            pred_ori, pred = getNearestTwoFeatureFromAllDB(emb, dbs, names)
            
            LOG.logI("label : {}".format(name[idx]))
            LOG.logI("pred : {}".format(pred))
            if (total+idx) % 10000 == 0 and (total+idx) != 0:
                LOG.logI("Done {} img...".format(total+idx))
            report.add(int(name[idx]), int(pred))
            print("{} {} {} {}".format(path[idx], name[idx], pred_ori, pred))
            f.write("{} {} {} {}\n".format(path[idx], name[idx], pred_ori, pred))

        total += db.shape
        LOG.logI("Total is {} now...".format(total))
    f.close()

    return report


def test(deepvac_config):
    config_cls = deepvac_config.cls
    dbs = []
    names = []
    paths = []

    db_paths = config_cls.db_paths
    map_locs = config_cls.map_locs
    np_paths = config_cls.np_paths
    for i in range(len(db_paths)):
        dbs.append(torch.load(db_paths[i], map_location = map_locs[i]))
        np_f = np.load(np_paths[i])
        names.append(np_f['names'])
        paths.append(np_f['paths'])

    report = getClassifierReport(dbs, names, paths, config_cls.file_path, config_cls.cls_num)
    report()

if __name__ == "__main__":
    #config_cls = deepvac_config.cls
    #report = compareAndReport(config_cls.feature_name, config_cls.file_path, config_cls.cls_num)
    #report()
    test(deepvac_config)

