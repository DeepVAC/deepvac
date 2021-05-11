from ..utils import LOG

def makeDivisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def separateBN4OptimizerPG(modules):
    paras_only_bn = []
    paras_wo_bn = []
    memo = set()
    gemfield_set = set()
    gemfield_set.update(set(modules.parameters()))
    LOG.logI("separateBN4OptimizerPG set len: {}".format(len(gemfield_set)))
    named_modules = modules.named_modules(prefix='')
    for module_prefix, module in named_modules:
        if "module" not in module_prefix:
            LOG.logI("separateBN4OptimizerPG skip {}".format(module_prefix))
            continue

        members = module._parameters.items()
        for k, v in members:
            name = module_prefix + ('.' if module_prefix else '') + k
            if v is None:
                continue
            if v in memo:
                continue
            memo.add(v)
            if "batchnorm" in str(module.__class__):
                paras_only_bn.append(v)
            else:
                paras_wo_bn.append(v)

    LOG.logI("separateBN4OptimizerPG param len: {} - {}".format(len(paras_wo_bn),len(paras_only_bn)))
    return paras_only_bn, paras_wo_bn




