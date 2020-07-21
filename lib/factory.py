import sys

class SyszuxFactory(object):
    def __init__(self):
        self.factory_dict = dict()
        self.start_prefix = ''
        self.end_suffix = ''
        self.omit_list = []
        self.configure()
        self.addProducts()

    def configure(self):
        raise Exception('Not implemented...')

    def addProduct(self, name, ins):
        self.factory_dict[name] = ins

    def initSym(self):
        self.sym = dir(self.product_kind)

    def omitUnderScoreSym(self):
        self.sym = [x for x in self.sym if not x.startswith('_')]

    def selectStartPrefix(self):
        self.sym = [x for x in self.sym if x.startswith(self.start_prefix)]

    def selectEndSuffix(self):
        self.sym = [x for x in self.sym if x.endswith(self.end_suffix)]

    def omitOmitList(self):
        self.sym = [x for x in self.sym if x not in self.omit_list]

    def getProducts(self):
        self.initSym()
        self.omitUnderScoreSym()
        self.selectStartPrefix()
        self.selectEndSuffix()
        self.omitOmitList()
        return self.sym

    def addProducts(self):
        raise Exception('Not implemented.')

    def get(self, ins_name):
        try:
            x = self.factory_dict[ins_name]
        except:
            print("ERROR! {} not found in factory.".format(ins_name))
            sys.exit(0)
        return x