import os
import numpy as np
import torch
import time
import faiss
from deepvac.syszux_deepvac import Deepvac
from deepvac.syszux_report import ClassifierReport
from deepvac.syszux_log import LOG

def getNearestTwoFeaturesFromAllDB(gpu_index, embs, dbs, emb_indexes, paths, k):
    D = I = []
    for i, db in enumerate(dbs):
        emb_index = emb_indexes[i]
        gpu_index.reset()
        gpu_index.add(db)
        tempD, tempI = gpu_index.search(embs, k)
        
        for nx in range(tempI.shape[0]):
            tempI[nx][0] = emb_index[tempI[nx][0]]
            tempI[nx][1] = emb_index[tempI[nx][1]]

        if len(D) == 0:
            D = tempD
            I = tempI
            continue
        D = np.c_[D, tempD]
        I = np.c_[I, tempI]
    
    return D, I

def getClassifierByFaissReport(dbs, emb_indexes, paths, filename, cls_num):
    
    report = ClassifierReport('gemfield',cls_num, cls_num)
    ngpus = faiss.get_num_gpus()

    d = 512
    LOG.logI('Load index start...')
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    LOG.logI('Load index finished...')
    f = open(filename, 'w')
    distance = index = []
    for idx, db in enumerate(dbs):
        LOG.logI('Process {} db...'.format(idx))
        emb_index = emb_indexes[idx]
        path = paths[idx]
        D, I = getNearestTwoFeaturesFromAllDB(gpu_index, db, dbs, emb_indexes, paths, 2)

        for n, dis in enumerate(D):
            dis_ = dis
            min_pos = np.argmin(dis)
            min_index = I[n][min_pos]
            dis_[np.argmin(dis)] = np.max(dis)
            sec_min_pos = np.argmin(dis_)
            sec_min_index = I[n][sec_min_pos]
            f.write('{} {} {} {}\n'.format(path[n], emb_index[n], min_index, sec_min_index))
            report.add(int(emb_index[n]), int(sec_min_index))
            #LOG.logI('path: {}; label: {}; nearest: {}; second nearest: {}\n'.format(path[n], emb_index[n], min_index, sec_min_index))
        LOG.logI('Process {} db finished...'.format(idx))

    
    f.close()
    return report

def test(config):
    db_paths = config.db_paths
    np_paths = config.np_paths
    assert len(db_paths) == len(np_paths)
    dbs = []
    emb_indexes = []
    paths = []

    for i in range(len(db_paths)):
        dbs.append(torch.load(db_paths[i]).to('cpu').numpy().astype('float32'))
        np_f = np.load(np_paths[i])
        emb_indexes.append(np_f['names'])
        paths.append(np_f['paths'])

    report = getClassifierByFaissReport(dbs, emb_indexes, paths, './report_info.txt', 90219)
    report()

if __name__ == "__main__":
    from config import config as deepvac_config
    test(deepvac_config.cls_faiss)
