from collections import OrderedDict
import time

class FaceReport(object):
    def __init__(self, ds_name = "Unknown", total_num = 0):
        self.total_num = total_num
        self.ds_name = ds_name
        self.initNumeratorKeys()
        self.initFuncDict()
        self.initReportFormat()
        self.refreshData()

    def initReportFormat(self):
        self.report_str = '''
|{}|{}|{}|{}|{}|{}|{}|{}|
|--|--|--|--|--|--|--|--|
|{}|{}|{}|{}|{}|{}|{}|{}|
        '''
    def initNumeratorKeys(self):
        self.denominator_keys = dict()
        self.numerator_keys = dict()
        self.denominator_keys['accuracy'] = self.denominator_keys['error'] = ['tp','fn_err', 'fn_none','fp','tn']
        self.denominator_keys['precision'] = ['tp','fp']
        self.denominator_keys['recall'] = self.denominator_keys['miss'] = ['tp', 'fn_err', 'fn_none']
        self.numerator_keys['accuracy'] = ['tp','tn']
        self.numerator_keys['precision'] = self.numerator_keys['recall'] = ['tp']
        self.numerator_keys['miss'] = ['fn_none']
        self.numerator_keys['error'] = ['fn_err','fn_none','fp']
    
    def initFuncDict(self):
        self.func_dict = dict()
        #TN,false -> false = 将负类预测为负类数；也即没注册db的ds 没有匹配 任何db；
        self.func_dict[(False,False)] = lambda gt,result : self.metrics.update({'tn':self.metrics['tn'] + 1})
        #FN,true -> false = 将正类预测为负类数；也即已注册db的ds 没有匹配 对应的db；
        self.func_dict[(True,False)] = lambda gt,result : self.metrics.update({'fn_none':self.metrics['fn_none'] + 1})
        #FP, false -> true = 将负类预测为正类数；也即没注册db的ds 匹配到了 某个db；
        self.func_dict[(False,True)] = lambda gt,result : self.metrics.update({'fp':self.metrics['fp'] + 1})
        #TP: true -> true = 将正类预测为正类数，也即已注册db的ds 匹配到了 对应的db；
        #或者 true -> true = 将正类预测为负类数；也即已注册db的ds 匹配到了 错误的db；
        self.func_dict[(True,True)] = lambda gt,result : self.metrics.update({'tp':self.metrics['tp'] + 1}) if gt == result else self.metrics.update({'fn_err':self.metrics['fn_err'] + 1})

    def initReportDict(self):
        self.report_dict = OrderedDict()
        self.report_dict['dataset'] = self.ds_name
        self.report_dict['total'] = self.total_num
        self.report_dict['duration'] = 0
        self.report_dict['accuracy'] = 0
        self.report_dict['precision'] = 0
        self.report_dict['recall'] = 0
        self.report_dict['miss'] = 0
        self.report_dict['error'] = 0

    def initMetricsDict(self):
        self.metric_keys = ['tp','fn_none','fp','tn','fn_err']
        self.metrics = dict.fromkeys(self.metric_keys)
        for k in self.metrics:
            self.metrics[k] = 0

    def refreshData(self):
        self.process_num = 0
        self.start_time = time.time()
        self.initMetricsDict()
        self.initReportDict()
    
    def reset(self):
        self.refreshData()
    
    def add(self, gt, result):
        """
        第一个参数gt为groundtruth 的label；如果要测试的ds图片并没有注册db，则gt为None；
            否则gt为对应的str(e.g. "famous_000001");
        第二个参数result为预测出来的label；如果要测试的ds图片没有匹配到任何db，则result为None；
            否则result为对应的str(e.g. "famous_000001");
        """
        self.process_num += 1

        k = (bool(gt), bool(result))
        self.func_dict[k](gt, result)

        assert sum(self.metrics.values()) == self.process_num, "internal error"
        assert self.process_num <= self.total_num, "total num error"
        return self

    def __call__(self):
        self.report_dict['duration'] = "%.3f"%(time.time() - self.start_time)

        f = lambda l: sum([org for gemfield, org in self.metrics.items() if gemfield in l])
        ff = lambda k: (f(self.numerator_keys[k]) / f(self.denominator_keys[k]) ) if f(self.denominator_keys[k]) != 0 else 0 

        report_keys = [k for k,v in self.report_dict.items()]
        report_values = [ff(k) if not v else v for k,v in self.report_dict.items() ]

        print(self.report_str.format(*report_keys, *report_values))

        assert self.process_num == self.total_num, "total num error"

if __name__ == "__main__":
    report = Report('gemfield',5)
    report.add("","2").add("1","1").add("1","1").add("1","1").add("1","1")
    report()






