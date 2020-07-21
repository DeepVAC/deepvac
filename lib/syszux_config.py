class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

config = AttrDict()
config.synthesis = AttrDict()
config.aug = AttrDict()
config.loader = AttrDict()
config.text = AttrDict()
config.detect = AttrDict()
config.face = AttrDict()
config.dumpimg = AttrDict()
