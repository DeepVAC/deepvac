import torch
from torch.quantization.fuser_method_mappings import DEFAULT_OP_LIST_TO_FUSER_METHOD
from torch.quantization import QuantStub, DeQuantStub

def get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def get_fuser_module_index(mod_list):
    rc = []
    if len(mod_list) < 2:
        return rc
    keys = sorted(list(DEFAULT_OP_LIST_TO_FUSER_METHOD.keys()), key=lambda x: len(x), reverse=True)
    mod2fused_list = [list(x) for x in keys]

    for mod2fused in mod2fused_list:
        if len(mod2fused) > len(mod_list):
            continue
        mod2fused_idx = [(i, i+len(mod2fused)) for i in range(len(mod_list) - len(mod2fused) + 1) if mod_list[i:i+len(mod2fused)] == mod2fused]
        if not mod2fused_idx:
            continue

        for idx in mod2fused_idx:
            start,end = idx
            mod_list[start: end] = [None] * len(mod2fused)

        rc.extend(mod2fused_idx)

    return rc

def auto_fuse_model(model):
    module_names = []
    module_types = []
    for name, m in model.named_modules():
        module_names.append(name)
        module_types.append(type(m))

    if len(module_types) < 2:
        return model

    module_idxs = get_fuser_module_index(module_types)
    modules_to_fuse = [module_names[mi[0]:mi[1]] for mi in module_idxs]
    new_model = torch.quantization.fuse_modules(model, modules_to_fuse)
    return new_model

class DeQuantStub(nn.Module):
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x: Any) -> Any:
        return x

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for sample, target in data_loader:
            model(sample)

def prepareQAT(deepvac_train_config):
    if not deepvac_train_config.qat_dir:
        return

    if deepvac_train_config.is_forward_only:
        LOG.logI("You are in forward_only mode, omit the parepareQAT()")
        return

    LOG.logI("You have enabled QAT, this step is only for prepare.")

    if deepvac_train_config.qat_net_prepared:
        LOG.logE("Error: You have already prepared the model for QAT.", exit=True)

    backend = 'fbgemm'
    if deepvac_train_config.quantize_backend:
        backend = deepvac_train_config.quantize_backend
    deepvac_train_config.qat_net_prepared = DeepvacQAT(deepvac_train_config.net).to(deepvac_train_config.device)
    deepvac_train_config.qat_net_prepared.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare_qat(deepvac_train_config.qat_net_prepared, inplace=True)
    #after this, train net will be transfered to QAT !
    deepvac_train_config.net = deepvac_train_config.qat_net_prepared

class DeepvacQAT(torch.nn.Module):
    def __init__(self, net2qat):
        super(DeepvacQAT, self).__init__()
        self.quant = QuantStub()
        self.net2qat = auto_fuse_model(net2qat)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.net2qat(x)
        x = self.dequant(x)
        return x