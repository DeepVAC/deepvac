from deepvac.syszux_feature_vector import NamesPathsClsFeatureVector, NamesPathsClsFeatureVectorByFaiss, NamesPathsClsFeatureVectorByFaissPytorch


if __name__ == "__main__":
    from config import config as deepvac_config

    # foundamental feature vector test
    fv = NamesPathsClsFeatureVector(deepvac_config.cls)
    fv.loadDB()
    fv.printClassifierReport()
    
    # faiss feature vector test
    fv = NamesPathsClsFeatureVectorByFaiss(deepvac_config.cls_faiss)
    fv.loadDB()
    fv.printClassifierReport()

    # faiss pytorch feature vector test
    fv = NamesPathsClsFeatureVectorByFaissPytorch(deepvac_config.cls_faiss_pth)
    fv.loadDB()
    fv.printClassifierReport()
