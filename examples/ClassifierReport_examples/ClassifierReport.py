import sys
sys.path.append('lib')
sys.path.append('../lib')
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
#       feature_name: name of feature files(4 pieces)
#       file_path: result file
#       cls_num: number of ids
# output:
#       report: report info, use `report()` to call
def compareAndReport(feature_name, file_path, cls_num):

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
            report.add(label, pred)
            f.write(inf)

    db1 = torch.load('db/{}_1.feature'.format(feature_name), map_location = {'cuda:0':'cuda:1'})
    db2 = torch.load('db/{}_2.feature'.format(feature_name), map_location = {'cuda:0':'cuda:1'})
    db3 = torch.load('db/{}_3.feature'.format(feature_name), map_location = {'cuda:1':'cuda:0'})
    db4 = torch.load('db/{}_4.feature'.format(feature_name), map_location = {'cuda:1':'cuda:0'})
    face_db1 = np.load('db/{}_1.feature.npz'.format(feature_name))
    face_db2 = np.load('db/{}_2.feature.npz'.format(feature_name))
    face_db3 = np.load('db/{}_3.feature.npz'.format(feature_name))
    face_db4 = np.load('db/{}_4.feature.npz'.format(feature_name))
    
    dbs = [db1, db2, db3, db4]
    names = [face_db1['names'], face_db2['names'], face_db3['names'], face_db4['names']]
    paths = [face_db1['paths'], face_db2['paths'], face_db3['paths'], face_db4['paths']]
    total = 0

    for i, cur_db in enumerate(dbs):
        name = names[i]
        path = paths[i]
        for idx, emb in enumerate(cur_db):
            start = time.time()
            if path[idx] in done_paths:
                continue
            LOG.log(LOG.S.I, "continue time : {}".format(time.time()-start))
            min_distance = []
            min_distance = getMinTup(min_distance, dbs[0], emb, names[0])
            min_distance = getMinTup(min_distance, dbs[1], emb, names[1])
                
            emb = emb.to('cuda:0')

            min_distance = getMinTup(min_distance, dbs[2], emb, names[2])
            min_distance = getMinTup(min_distance, dbs[3], emb, names[3])

            sorted_min_distance = sorted(min_distance, key=lambda t:t[1])

            pred = sorted_min_distance[1][0]
            pred_ori = sorted_min_distance[0][0]
            #print('label : {}'.format(name[idx]))
            #print('pred : {}'.format(pred))
            LOG.log(LOG.S.I, "label : {}".format(name[idx]))
            LOG.log(LOG.S.I, "pred : {}".format(pred))
            if (total+idx) % 10000 == 0 and (total+idx) != 0:
                LOG.log(LOG.S.I, "Done {} img...".format(total+idx))
            report.add(int(name[idx]), int(pred))
            print("{} {} {} {}".format(path[idx], name[idx], pred_ori, pred))
            f.write("{} {} {} {}\n".format(path[idx], name[idx], pred_ori, pred))

        total += cur_db.shape
        LOG.log(LOG.S.I, "Total is {} now...".format(total))
    f.close()

    return report


if __name__ == "__main__":

    config_cls = deepvac_config.cls
    report = face.compareAndReport(config_cls.feature_name, config_cls.file_path, config_cls.cls_num)
    report()

