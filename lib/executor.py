import pprint
import sys
sys.path.append('lib')
from collections import defaultdict
from syszux_aug_factory import AugFactory

class Chain(object):
    def __init__(self, flow_list):
        flow = flow_list.split("=>")
        self.op_sym_list = [x.strip() for x in flow]
        self.op_sym_list = [x.strip() for x in self.op_sym_list if x]
        self.op_list = []

    def __call__(self):
        for t in self.op_list:
            t()
    
    def addOp(self, op):
        self.op_list.append(op)

class ProductChain(Chain):
    def __init__(self, flow_list):
        super(ProductChain, self).__init__(flow_list)

    def __call__(self, img):
        for t in self.op_list:
            img = t(img)
        return img
    
    def addOp(self, op):
        self.op_list.append(op)

class DeepvacChain(ProductChain):
    def __init__(self, flow_list, deepvac_config):
        super(DeepvacChain, self).__init__(flow_list)
        self.op_list = [eval("DeepvacChain.{}".format(x))(deepvac_config) for x in self.op_sym_list]

class AugChain(ProductChain):
    def __init__(self, flow_list, deepvac_config):
        super(AugChain, self).__init__(flow_list)
        self.factory = AugFactory()
        self.op_list = [self.factory.get(x)(deepvac_config) for x in self.op_sym_list]
        
class Executor(object):
    def __init__(self, edges):
        self._graph = defaultdict(set)
        self.makeConnections(edges)

    def makeConnections(self, edges):
        for node1, node2 in edges:
            self.addEdge(node1, node2)

    def addEdge(self, node1, node2):
        self._graph[node1].add(node2)

    def remove(self, node):
        for k, set_v in self._graph.items():
            try:
                set_v.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def __call__(self, node):
        pass

    def dumpDag(self):
        pretty_print = pprint.PrettyPrinter()
        pretty_print.pprint(self._graph)

if __name__ == "__main__":
    edges = [('gemfield', 'leaflower'), ('leaflower', 'civilnet'), ('civilnet', 'syszux'),
                   ('syszux', 'deepvac'), ('deepvac', 'syszux')]
    
    e = Executor(edges)
    e.dumpDag()
    e.remove('civilnet')
    e.dumpDag()
    