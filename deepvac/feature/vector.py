import os
import sys
import numpy as np
import torch
import time
try:
    import faiss
except:
    LOG.W("Faiss not installed. You cannot use the API implemented based on Faiss library.")

from deepvac.syszux_report import ClassifierReport
from deepvac.syszux_log import LOG

def swigPtrFromTensor(x):
    """ gets a Faiss SWIG pointer from a pytorch trensor (on CPU or GPU) """
    assert x.is_contiguous()

    if x.dtype == torch.float32:
        return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)
    
    if x.dtype == torch.int64:
        return faiss.cast_integer_to_long_ptr(x.storage().data_ptr() + x.storage_offset() * 8)

    raise Exception("tensor type not supported: {}".format(x.dtype))

#from config import config
#config.feature as feature_vector_config
class FeatureVector(object):
    def __init__(self, feature_vector_config):
        self.config = feature_vector_config
        assert self.config.db_path_list, 'You must configure config.db_path_list in config.py'
        assert self.config.map_locs, 'You must configure config.map_locs in config.py, which is to specify how mapping the feature to CUDA/CPU device'
        assert len(self.config.map_locs) == len(self.config.db_path_list), 'config.db_path_list number should be same with config.map_locs'
        self.dbs = []
        if not self.config.name:
            self.config.name = 'gemfield'

        if not self.config.log_every:
            self.config.log_every = 10000

    def loadDB(self):
        for db_path, map_loc in zip(self.config.db_path_list, self.config.map_locs):
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
    def __init__(self, feature_vector_config):
        super(NamesPathsClsFeatureVector, self).__init__(feature_vector_config)
        assert self.config.class_num, 'You must configure config.class_num in config.py'
        assert self.config.np_path_list, 'You must configure config.np_path_list in config.py'
        assert len(self.config.np_path_list) == len(self.config.db_path_list), 'config.db_path_list number should be same with config.np_path_list'
        self.names = []
        self.paths = []

    def loadDB(self):
        super(NamesPathsClsFeatureVector, self).loadDB()
        for np_path in self.config.np_path_list:
            npf = np.load(np_path)
            self.names.append(npf['names'])
            self.paths.append(npf['paths'])

    def searchAllDB(self, xq, k=2):
        TMP_D = []
        TMP_N = []
        for i, db in enumerate(self.dbs):
            xq = xq.to(db.device)
            distances, indices = self.searchDB(db, xq, k)
            TMP_D.extend(distances)
            TMP_N.extend(self.names[i][indices])

        N = [n for _,n in sorted(zip(TMP_D,TMP_N))]
        D = sorted(TMP_D)

        return D[:k], N[:k]

    def printClassifierReport(self):
        report = ClassifierReport(self.config.name, self.config.class_num, self.config.class_num)
        for i, db in enumerate(self.dbs):
            name = self.names[i]
            path = self.paths[i]
            for idx, emb in enumerate(db):
                _, N = self.searchAllDB(emb)
                should_be_gt, pred = N[0], N[1]
                LOG.logI("label : {}".format(name[idx]))
                LOG.logI("pred : {}".format(pred))
                if idx % self.config.log_every == 0:
                    LOG.logI("Process {} img...".format(idx))
                report.add(int(name[idx]), int(pred))
        report()

class NamesPathsClsFeatureVectorByFaiss(object):
    def __init__(self, feature_vector_config):
        if 'faiss' not in sys.modules:
            LOG.logE("Faiss not installed, cannot use this class.", exit=True)
        self.config = feature_vector_config
        assert self.config.db_path_list, 'You must configure config.db_path_list in config.py'
        assert self.config.np_path_list, 'You must configure config.np_path_list in config.py'
        assert len(self.config.np_path_list) == len(self.config.db_path_list), 'config.np_path_list number should be same with config.db_path_list'
                        
        if not self.config.name:
            self.config.name = 'gemfield'
        if not self.config.log_every:
            self.config.log_every = 10000
        
        self.dbs = []
        self.names = []
        self.paths = []
        self.gpu_index = []

    def loadDB(self):
        for db_path, np_path in zip(self.config.db_path_list, self.config.np_path_list):
            npf = np.load(np_path)
            self.dbs = np.vstack((self.dbs, torch.load(db_path).to('cpu').numpy().astype('float32'))) \
                    if len(self.dbs) else torch.load(db_path).to('cpu').numpy().astype('float32')
            self.names = np.hstack((self.names, npf['names'])) if len(self.names) else npf['names']
            self.paths = np.hstack((self.paths, npf['paths'])) if len(self.paths) else npf['paths']
        torch.cuda.empty_cache()
        self.__loadIndex()

    def __loadIndex(self):
        assert self.dbs != [], "You should load db before load index, use self.loadDB() ..."
        d = self.dbs[0].shape[-1]
        ngpu = faiss.get_num_gpus()
        index = faiss.IndexFlatL2(d)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        gpu_resources = []
        
        for i in range(0, ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
            vdev.push_back(i)
            vres.push_back(res)
        
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        self.gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        self.gpu_index.referenced_objects = gpu_resources
        self.gpu_index.add(self.dbs)

    def searchAllDB(self, xq, k=2):
        assert self.gpu_index != [], "You should load index before search, use self.loadIndex() ..."
        D, I = self.gpu_index.search(xq, k)
        N = self.names[I]

        return D, N

    def printClassifierReport(self):
        report = ClassifierReport(self.config.name, self.config.class_num, self.config.class_num)
        RD, RN = self.searchAllDB(self.dbs)

        for idx, n in enumerate(RN):
            pred_ori, pred = n[0], n[1]
            LOG.logI("label : {}".format(self.names[idx]))
            LOG.logI("pred : {}".format(pred))
            if idx % self.config.log_every == 0:
                LOG.logI("Process {} img...".format(idx))
            
            report.add(int(self.names[idx]), int(pred))
        report()

class NamesPathsClsFeatureVectorByFaissMulBlock(NamesPathsClsFeatureVectorByFaiss):
    def __init__(self, feature_vector_config):
        super(NamesPathsClsFeatureVectorByFaissMulBlock, self).__init__(feature_vector_config)
        assert len(self.config.np_path_list) == len(self.config.block_map), "config.np_path_list number should be same with config.block_map"
        self.dbs = [[]] * faiss.get_num_gpus()
        self.names = [[]] * faiss.get_num_gpus()
        self.paths = [[]] * faiss.get_num_gpus()

    def loadDB(self):
        for block_map, db_path, np_path in zip(self.config.block_map, self.config.db_path_list, self.config.np_path_list):
            npf = np.load(np_path)
            self.dbs[block_map] = np.vstack((self.dbs[block_map], torch.load(db_path).to('cpu').numpy().astype('float32'))) \
                                    if len(self.dbs[block_map]) else torch.load(db_path).to('cpu').numpy().astype('float32')
            self.names[block_map] = np.hstack((self.names[block_map], npf['names'])) if len(self.names[block_map]) else npf['names']
            self.paths[block_map] = np.hstack((self.paths[block_map], npf['paths'])) if len(self.paths[block_map]) else npf['paths']
        
        
        torch.cuda.empty_cache()
        self.__loadIndex()

    def __loadIndex(self):
        assert self.dbs != [], "You should load db before load index, use self.loadDB() ..."
        d = self.dbs[0].shape[-1]
        ngpu = faiss.get_num_gpus()
        index = faiss.IndexFlatL2(d)

        res = faiss.StandardGpuResources()
        
        for i, db in enumerate(self.dbs):
            gpu_index = faiss.index_cpu_to_gpu(res, i, index)
            gpu_index.add(db)
            self.gpu_index.append(gpu_index)

    def searchAllDB(self, xq, k=2):
        assert self.gpu_index != [], "You should load index before search, use self.loadIndex() ..."
        
        D = []
        N = []
        for i, gindex in enumerate(self.gpu_index):
            tempD, tempI = gindex.search(xq, k)
            tempN = self.names[i][tempI]
            if i == 0:
                D, N = tempD, tempN
                continue
            D = np.hstack((D, tempD))
            N = np.hstack((N, tempN))

        RD, RN = D, N
        for i, d in enumerate(D):
            index = np.argsort(d)
            RD[i] = D[i][index]
            RN[i] = N[i][index]
            
        return RD[:, :k], RN[:, :k]

    def printClassifierReport(self):
        report = ClassifierReport(self.config.name, self.config.class_num, self.config.class_num)
        
        for i, db in enumerate(self.dbs):
            name = self.names[i]
            RD, RN = self.searchAllDB(db)

            for idx, n in enumerate(RN):
                pred_ori, pred = n[0], n[1]
                LOG.logI("label : {}".format(name[idx]))
                LOG.logI("pred : {}".format(pred))
                if idx % self.config.log_every == 0:
                    LOG.logI("Process {} img...".format(idx))
                
                report.add(int(name[idx]), int(pred))
            LOG.logI("{} xq finished...".format(i))
                
        report()

class NamesPathsClsFeatureVectorByFaissPytorch(NamesPathsClsFeatureVector):
    def __init__(self, feature_vector_config):
        super(NamesPathsClsFeatureVectorByFaissPytorch, self).__init__(feature_vector_config)
        self.dbs = []
                           
    def loadDB(self):
        super(NamesPathsClsFeatureVectorByFaissPytorch, self).loadDB()
        assert self.config.db_path_list, 'You must configure config.db_path_list in config.py'
        assert self.config.map_locs, 'You must configure config.map_locs in config.py'
        for db_path, map_loc in zip(self.config.db_path_list, self.config.map_locs):
            self.dbs.append(torch.load(db_path, map_location = map_loc))

    def search_index_pytorch(self, index, x, k, D=None, I=None):
        """call the search function of an index with pytorch tensor I/O (CPU
            and GPU supported)"""
        assert x.is_contiguous()
        n, d = x.size()
        assert d == index.d
        
        if D is None:
            D = torch.empty((n, k), dtype=torch.float32, device=x.device)             
        else:                                        
            assert D.size() == (n, k)
        
        if I is None:
            I = torch.empty((n, k), dtype=torch.int64, device=x.device)             
        else:                                                           
            assert I.size() == (n, k)
        
        torch.cuda.synchronize()
        xptr = swigPtrFromTensor(x)
        Iptr = swigPtrFromTensor(I)
        Dptr = swigPtrFromTensor(D)
        index.search_c(n, xptr, k, Dptr, Iptr)
        torch.cuda.synchronize()
        
        return D, I

    def searchAllDB(self, xq, k=2):        
        D = []
        N = []
        d = 512
        LOG.logI('Load index start...')
        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatL2(res, d)
        LOG.logI('Load index finished...')

        for i, db in enumerate(self.dbs):
            name = self.names[i]
            gpu_index.reset()
            gpu_index.add(db.to('cpu').numpy().astype('float32'))
            tempD, tempI = self.search_index_pytorch(gpu_index, xq, k)
            tempD = tempD.to('cpu').numpy()
            tempI = tempI.to('cpu').numpy()
            tempN = name[tempI]
            
            if len(D) == 0:
                D, N = tempD, tempN
                continue

            D = np.hstack((D, tempD))
            N = np.hstack((N, tempN))
            
        RD, RN = D, N    
        for i, d in enumerate(D):
            index = np.argsort(d)
            RD[i] = D[i][index]
            RN[i] = N[i][index]
        
        return RD[:, :k], RN[:, :k]

    def printClassifierReport(self):        
        report = ClassifierReport(self.config.name, self.config.class_num, self.config.class_num)
        for i, xq in enumerate(self.dbs):
            LOG.logI('Process {} db...'.format(i))
            name = self.names[i]
            path = self.paths[i]
            RD, RN = self.searchAllDB(xq)
            
            for idx, n in enumerate(RN):
                pred_ori, pred = n[0], n[1]
                LOG.logI("label : {}".format(name[idx]))
                LOG.logI("pred : {}".format(pred))
                if idx % self.config.log_every == 0:
                    LOG.logI("Process {} img...".format(idx))
                
                report.add(int(name[idx]), int(pred))
        
        report()

if __name__ == "__main__":
    from config import config
    fv = NamesPathsClsFeatureVector(config)
    fv.printClassifierReport()

def loadDB(self, db_path):
    self.xb = torch.load(db_path).to(self.device)

def addEmb2DB(self, emb):
    self.xb = torch.cat((self.xb, emb))

def saveDB(self, db_path):
    torch.save(self.xb, db_path)

def search(self, xq, k=1):
    D = []
    I = []
    if k < 1 or k > 10:
        LOG.logE('illegal nearest neighbors parameter k(1 ~ 10): {}'.format(k))
        return D, I

    distance = torch.norm(self.xb - xq, dim=1)
    for i in range(k):
        values, indices = distance.kthvalue(i+1)
        D.append(values.item())
        I.append(indices.item())
    return D, I