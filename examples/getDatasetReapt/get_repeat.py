import re
from collections import Counter

def getAllPairs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    pairs = []
    p1 = re.compile(r'[[](.*?)[]]', re.S)
    p2 = re.compile(r'[(](.*?)[)]')
    for line in lines:
        label, line = line.split(':')
        res = re.findall(p1, line.strip())
        tups = re.findall(p2, res[0])
        for tup in tups:
            pred, scale = tup.split(', ')
            pairs.append((label, pred, scale))
    return pairs

def isUseful(pair):
    label_idx = int(pair[0].split('/')[1])
    pred_idx = int(pair[1].split('/')[1])
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

def getIndexList(pairs):
    indexes = []
    for pair in pairs:
        indexes.append(pair[0])
        indexes.append(pair[1])
    return indexes

def delPairs(pairs, max_idx):
    new_pairs = []
    for pair in pairs:
        if pair[0] == max_idx or pair[1] == max_idx:
            continue
        new_pairs.append(pair)
    return new_pairs

def genConflictPairs(pairs):
    new_pairs = pairs
    delete = []
    while True:
        indexes = getIndexList(new_pairs)
        res = Counter(indexes)
        if len(res) == 0 or res.most_common(1)[0][1] <= 1:
            break
        max_idx = res.most_common(1)[0][0]
        delete.append(max_idx)
        new_pairs = delPairs(new_pairs, max_idx)

    return new_pairs, delete

def getConflictChecklist(input_path, output_path):
    pairs = getAllPairs(input_path)
    output_f = open(output_path, 'w')
    filter_pairs = []
    delete = []
    for pair in pairs:
        if not isUseful(pair):
            continue
        filter_pairs.append(pair)
    
    conflict_pairs, delete = genConflictPairs(filter_pairs)

    for pair in conflict_pairs:
        label_idx = int(pair[0].split('/')[1])
        pred_idx = int(pair[1].split('/')[1])
        del_ = pair[0] if label_idx >= pred_idx else pair[1]
        delete.append(del_)
    
    for d in delete:
        output_f.write('{}\n'.format(d))

    output_f.close()

            

if __name__ == "__main__":
    from config import config as deepvac_config
    config_repeat = deepvac_config.repeat
    getConflictChecklist(config_repeat.input_path, config_repeat.output_path)
