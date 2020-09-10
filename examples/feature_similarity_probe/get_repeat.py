import re
from collections import Counter

class FeatureSimilarityProbe(object):
    def __init__(self, deepvac_config):
        self.conf = deepvac_config
        assert self.conf.input_path, "You must configure config.input_path in config.py"
        assert self.conf.output_path, "You must configure config.output_path in config.py"
        self.pairs = []
        self.delete = []

    def loadPairs(self):
        with open(self.conf.input_path, 'r') as f:
            lines = f.readlines()
        p1 = re.compile(r'[[](.*?)[]]', re.S)
        p2 = re.compile(r'[(](.*?)[)]')
        for line in lines:
            _, line = line.split(':')
            res = re.findall(p1, line.strip())
            tups = re.findall(p2, res[0])
            for tup in tups:
                label, pred, scale = tup.split(', ')
                self.pairs.append((label, pred, scale))

    def __isUseful(self, pair):
        label_idx = int(pair[0].split('/')[1]) if '/' in pair[0] else int(pair[0])
        pred_idx = int(pair[1].split('/')[1]) if '/' in pair[1] else int(pair[1])
        scale = float(pair[2])
        if scale >= 0.3:
            return True
        if scale >= 0.1 and abs(label_idx - pred_idx) <= 5:
            return True
        if scale >=0.02 and abs(label_idx - pred_idx) <= 2:
            return True
        if scale > 0. and abs(label_idx - pred_idx) <= 1:
            return True
        return False

    def __getIndexList(self):
        indexes = []
        for pair in self.pairs:
            indexes.append(pair[0])
            indexes.append(pair[1])
        return indexes

    def __genDeleteIdWithMoreOccurrence(self):
        while True:
            indexes = self.__getIndexList()
            res = Counter(indexes)
            if len(res) == 0 or res.most_common(1)[0][1] <= 1:
                break
            max_idx = res.most_common(1)[0][0]
            self.delete.append(max_idx)
            self.pairs = [pair for pair in self.pairs if pair[0] != max_idx and pair[1] != max_idx]

    def __genDeleteIdWithGreaterId(self):
        for pair in self.pairs:
            label_idx = int(pair[0].split('/')[1]) if '/' in pair[0] else pair[0]
            pred_idx = int(pair[1].split('/')[1]) if '/' in pair[1] else pair[1]
            del_ = pair[0] if label_idx >= pred_idx else pair[1]
            self.delete.append(del_)

    def dumpIdToDelete(self):
        self.pairs = [pair for pair in self.pairs if self.__isUseful(pair)]
        self.__genDeleteIdWithMoreOccurrence()
        self.__genDeleteIdWithGreaterId()

    def writeDeleteIdChecklistFile(self):
        self.dumpIdToDelete()
        with open(self.conf.output_path, 'w') as f:
            for d in self.delete:
                f.write('{}\n'.format(d))

if __name__ == "__main__":
    from config import config as deepvac_config
    fs = FeatureSimilarityProbe(deepvac_config.repeat)
    fs.loadPairs()
    fs.writeDeleteIdChecklistFile()
