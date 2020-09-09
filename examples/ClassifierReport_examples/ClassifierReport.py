import sys
sys.path.insert(0, '/gemfield/hostpv/wangyuhang/new/deepvac/')
from deepvac.syszux_feature_vector import NamesPathsClsFeatureVector, NamesPathsClsFeatureVectorByFaiss, NamesPathsClsFeatureVectorByFaissPytorch, NamesPathsClsFeatureVectorByFaissMulBlock


if __name__ == "__main__":
    from config import config as deepvac_config

    # foundamental feature vector test
    fv = NamesPathsClsFeatureVector(deepvac_config.cls)
    fv.loadDB()
    fv.printClassifierReport()
    
    # faiss feature vector test
    fv = NamesPathsClsFeatureVectorByFaiss(deepvac_config.cls_faiss)
    fv.loadDB()
    fv.loadIndex()
    fv.printClassifierReport()

    # faiss mul block feature vector test
    fv = NamesPathsClsFeatureVectorByFaissMulBlock(deepvac_config.cls_faiss_block)
    fv.loadDB()
    fv.loadIndex()
    fv.printClassifierReport()

    # faiss pytorch feature vector test
    fv = NamesPathsClsFeatureVectorByFaissPytorch(deepvac_config.cls_faiss_pth)
    fv.loadDB()
    fv.printClassifierReport()
