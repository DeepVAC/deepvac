import random
from collections import OrderedDict
from .factory import AugFactory

class Composer(object):
    def __init__(self,deepvac_config):
        self._graph = OrderedDict()
        self._graph_p = OrderedDict()
        self.config = deepvac_config
        self.auditConfig()

    def addAugFactory(self, name, chain, p=1.0):
        self._graph[name] = chain
        self._graph_p[name] = p

    def auditConfig(self):
        pass

    def addOp(self, name, op, p=1.0):
        self._graph[name].addOp(op,p)

    def remove(self, name):
        try:
            del self._graph[name]
        except KeyError:
            pass

    def __call__(self, img):
        for k in self._graph:
            if random.random() < self._graph_p[k]:
                img = self._graph[k](img)
        return img

class DiceComposer(Composer):
    def __call__(self, img):
        i = random.randrange(len(self._graph_p))
        return list(self._graph.values())[i](img)

class PickOneComposer(Composer):
    def __call__(self, img):
        for k in self._graph:
            if random.random() >= self._graph_p[k]:
                continue
            return self._graph[k](img)
        return self._graph[k](img)

class MultiInputCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class FaceAugComposer(Composer):
    def __init__(self, deepvac_config):
        super(FaceAugComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('RandomColorJitterAug@0.5 => MosaicAug@0.5',deepvac_config)
        ac2 = AugFactory('IPCFaceAug || MotionAug',deepvac_config)
        self.addAugFactory('ac1', ac1, 1)
        self.addAugFactory('ac2', ac2, 0.5)

class OcrAugComposer(Composer):
    def __init__(self, deepvac_config):
        super(OcrAugComposer, self).__init__(deepvac_config)

        ac1 = AugFactory('SpeckleAug@0.2 => HorlineAug@0.2 => NoisyAug@0.2',deepvac_config)
        ac2 = AugFactory('MotionAug || AffineAug || PerspectAug || GaussianAug || VerlineAug || LRmotionAug || UDmotionAug || PerspectiveAug || StretchAug',deepvac_config)

        self.addAugFactory('ac1', ac1, 0.2)
        self.addAugFactory('ac2', ac2, 0.9)

class YoloAugComposer(Composer):
    def __init__(self, deepvac_config):
        super(YoloAugComposer, self).__init__(deepvac_config)
        if deepvac_config.hflip is None:
            deepvac_config.hflip = 0.5
        ac = AugFactory("YoloPerspectiveAug => HSVAug => YoloNormalizeAug => YoloHFlipAug@{}".format(deepvac_config.hflip), deepvac_config)
        self.addAugFactory("ac", ac)

class RetinaAugComposer(PickOneComposer):
    def __init__(self, deepvac_config):
        super(RetinaAugComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('CropFacialWithBoxesAndLmksAug => BrightDistortFacialAug@0.5 => ContrastDistortFacialAug@0.5 => SaturationDistortFacialAug@0.5 \
                => HueDistortFacialAug@0.5 => Pad2SquareFacialAug => MirrorFacialAug@0.5 => ResizeSubtractMeanFacialAug', deepvac_config)
        ac2 = AugFactory('CropFacialWithBoxesAndLmksAug => BrightDistortFacialAug@0.5 => SaturationDistortFacialAug@0.5 => HueDistortFacialAug@0.5 \
        => ContrastDistortFacialAug@0.5 => Pad2SquareFacialAug => MirrorFacialAug@0.5 => ResizeSubtractMeanFacialAug', deepvac_config)
        self.addAugFactory("ac1", ac1, 0.5)
        self.addAugFactory("ac2", ac2, 0.5)

class OcrDetectAugComposer(Composer):
    def __init__(self, deepvac_config):
        super(OcrDetectAugComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('ImageWithMasksRandomHorizontalFlipAug@0.5 => ImageWithMasksRandomRotateAug => ImageWithMasksRandomCropAug',deepvac_config)
        self.addAugFactory('ac1', ac1, 1)

class SegImageAugComposer(Composer):
    def __init__(self, deepvac_config):
        super(SegImageAugComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('GaussianAug || RandomColorJitterAug || BrightnessJitterAug || ContrastJitterAug',deepvac_config)
        ac2 = AugFactory('MotionAug',deepvac_config)
        self.addAugFactory('ac1', ac1, 1)
        self.addAugFactory('ac2', ac1, 0.2)

class SegImageWithMaskAugComposer(Composer):
    def __init__(self, deepvac_config):
        super(SegImageWithMaskAugComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('ImageWithMasksRandomCropAug',deepvac_config)
        ac2 = AugFactory('ImageWithMasksRandomHorizontalFlipAug || ImageWithMasksRandomRotateAug',deepvac_config)
        self.addAugFactory('ac1', ac1, 0.4)
        self.addAugFactory('ac2', ac2, 0.6)
