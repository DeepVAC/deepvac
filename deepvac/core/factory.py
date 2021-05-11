from ..utils import LOG

class SyszuxFactory(object):
    def __init__(self):
        self.factory_dict = dict()
        self.locals = None
        self.auditConfig()
        self.initProducts()
        self.initMyProducts()

    def auditConfig(self):
        self.start_prefix = ''
        self.end_suffix = ''
        self.omit_list = []

    def initProducts(self):
        LOG.logE("You must reimplement initProducts() function.", exit=True)
    
    def initMyProducts(self):
        pass

    def addProduct(self, name, ins):
        LOG.logE("You must reimplement addProduct() function in subclass.", exit=True)

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

    def getSymsFromProduct(self):
        self.initSym()
        self.omitUnderScoreSym()
        self.selectStartPrefix()
        self.selectEndSuffix()
        self.omitOmitList()
        return self.sym

    def addProducts(self, product, product_name, start_prefix='', end_suffix='', omit_list=[]):
        if product_name not in self.locals:
            LOG.logE("{} not in locals(). You may forget set self.locals = locals in subclass auditConfig API.".format(product_name), exit=True)
            return
        self.product_kind = product
        self.omit_list = omit_list
        self.start_prefix = start_prefix
        self.end_suffix = end_suffix
        for sym in self.getSymsFromProduct():
            new_sym = sym
            if '_aug' not in product_name:
                new_sym = '{}.{}'.format(product_name, new_sym)
            self.addProduct(new_sym, '{}.{}'.format(product_name, sym) )

    def get(self, ins_name):
        try:
            x = self.factory_dict[ins_name]
        except:
            LOG.logE("ERROR! {} not found in factory.".format(ins_name), exit=True)
        return x