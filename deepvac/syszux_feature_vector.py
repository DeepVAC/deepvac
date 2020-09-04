from config import config as deepvac_config
import os
import numpy as np
import torch
import time
from deepvac.syszux_report import ClassifierReport
from deepvac.syszux_log import LOG

class FeatureVector(object):
    def __init__(deepvac_config):
        self.conf = deepvac_config
        assert self.conf.db_path_list, 'You must configure config.db_path_list in config.py'
        assert self.conf.map_locs, 'You must configure config.map_locs in config.py, which is to specify how mapping the feature to CUDA/CPU device'
        assert len(self.conf.map_locs) == len(self.conf.db_path_list), 'config.db_path_list number should be same with config.map_locs'
        self.dbs = []
        if not self.conf.name:
            self.conf.name = 'gemfield'

        if not self.conf.log_every:
            self.conf.log_every = 10000

    def loadDB(self):
        for db_path, map_loc in zip(self.conf.db_path_list, self.conf.map_locs):
            self.dbs.append(torch.load(db_path, map_location = map_loc))

    def searchDB(self, xb, xq, k=2):
        D = []
        I = []
        if k < 1 or k > 10:
            LOG.logE('illegal nearest neighbors parameter k(1 ~ 10): {}'.format(k))
            return D, I

        distance = torch.norm(xb - xq, dim=1)
        for i in range(k):
            values, indices = distance.kthvalue(i+1)
            D.append(values.item())
            I.append(indices.item())
        return D, I

    def searchAllDB(self, xq, k=2):
        raise Exception("Cannot do searchAllDB in base class, since no names can be used to re index feature accross multi DB BLOCK.")

    def printClassifierReport(self):
        pass

class NamesPathsClsFeatureVector(FeatureVector):
    def __init__(deepvac_config):
        super(NamesPathsClsFeatureVector, self).__init__(deepvac_config)
        assert self.conf.class_num, 'You must configure config.class_num in config.py'
        assert self.conf.np_path_list, 'You must configure config.np_path_list in config.py'
        self.names = []
        self.paths = []

    def loadDB(self):
        super(NamesPathsClsFeatureVector, self).loadDB()
        for np_path in self.conf.np_path_list:
            npf = np.load(np_path)
            self.names.append(npf['names'])
            self.paths.append(npf['paths'])

    def searchDB(self, xb, xq):
        D,I = super(NamesPathsClsFeatureVector, self).searchDB(xb, xq)
        if not D or not I:
            return D,I,[]
        return D,I,self.names[I]

    def searchAllDB(self, xq, k=2):
        D = []
        N = []
        for db in self.dbs:
            xq = xq.to(db.device)
            distances, indices, names = self.searchDB(db, xq, k)
            D.extend(distances)
            N.extend(names)

        RN = [n for _,n in sorted(zip(D,N))]
        RD = sorted(D)

        return RD[:k], RN[:k]

    def printClassifierReport(self):
        report = ClassifierReport(self.conf.name, self.conf.class_num, self.conf.class_num)
        for i, db in enumerate(self.dbs):
            name = self.names[i]
            path = paths[i]
            for idx, emb in enumerate(db):
                pred_ori, pred = self.searchAllDB(emb)
                LOG.logI("label : {}".format(name[idx]))
                LOG.logI("pred : {}".format(pred))
                if idx % self.conf.log_every == 0:
                    LOG.logI("Process {} img...".format(idx))
                report.add(int(name[idx]), int(pred))
        report()

if __name__ == "__main__":
    from config import config
    fv = NamesPathsClsFeatureVector(config)
    fv.printClassifierReport()

