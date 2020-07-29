from collections import OrderedDict
import time

class Report(object):
    def __init__(self, ds_name = "Unknown", total_num = 0):
        self.total_num = total_num
        self.ds_name = ds_name
        self.initNumeratorKeys()
        self.initFuncDict()
        self.initReportFormat()
        self.reset()

    def add(self, gt, predict):
        """
        第一个参数gt为groundtruth 的label；如果要测试的ds图片并没有注册db，则gt为None；
            否则gt为对应的str(e.g. "famous_000001");
        第二个参数predict为预测出来的label；如果要测试的ds图片没有匹配到任何db，则predict为None；
            否则predict为对应的str(e.g. "famous_000001");
        """
        self.process_num += 1

        k = (bool(gt), bool(predict))
        self.func_dict[k](gt, predict)

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

    def initReportFormat(self):
        self.report_str = '''
|{}|{}|{}|{}|{}|{}|{}|{}|
|--|--|--|--|--|--|--|--|
|{}|{}|{}|{}|{}|{}|{}|{}|
        '''
    def initMetricsDict(self):
        self.metric_keys = ['tp','tn','fp','fn']
        self.metrics = dict.fromkeys(self.metric_keys)
        for k in self.metrics:
            self.metrics[k] = 0

    def reset(self):
        self.process_num = 0
        self.start_time = time.time()
        self.initMetricsDict()
        self.initReportDict()

    def initNumeratorKeys(self):
        self.denominator_keys = dict()
        self.numerator_keys = dict()
        self.denominator_keys['accuracy'] = self.denominator_keys['miss'] = self.denominator_keys['error'] = ['tp','fp','tn','fn']
        self.denominator_keys['precision'] = ['tp','fp']
        self.denominator_keys['recall'] =  ['tp','fn']
        self.numerator_keys['accuracy'] = ['tp','tn']
        self.numerator_keys['precision'] = self.numerator_keys['recall'] = ['tp']
        self.numerator_keys['miss'] = ['fn']
        self.numerator_keys['error'] = ['fp','fn']

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

    def initFuncDict(self):
        self.func_dict = dict()
        #预测结果大于阈值，且正确的是TP，且错误的是FP；
        self.func_dict[(True,True)] = lambda gt,predict : self.metrics.update({'tp':self.metrics['tp'] + 1}) if gt == predict else self.metrics.update({'fp':self.metrics['fp'] + 1})
        #FP,预测结果大于阈值，且错误；
        self.func_dict[(False,True)] = lambda gt,predict : self.metrics.update({'fp':self.metrics['fp'] + 1})
        #TN,预测结果小于阈值，且正确；也即没注册db的ds 没有匹配 任何db；
        self.func_dict[(False,False)] = lambda gt,predict : self.metrics.update({'tn':self.metrics['tn'] + 1})
        #FN,预测结果小于阈值，且错误；也即已注册db的ds 没有匹配 任何db；
        self.func_dict[(True,False)] = lambda gt,predict : self.metrics.update({'fn':self.metrics['fn'] + 1})

class FaceReport(Report):
    def __init__(self, ds_name = "Unknown", total_num = 0):
        super(FaceReport,self).__init__(ds_name, total_num)

#For SYSZUXocr test dataset.
class OcrReport(Report):
    def __init__(self, ds_name = "Unknown", total_num = 0):
        super(OcrReport,self).__init__(ds_name, total_num)

    def reset(self):
        super(OcrReport, self).reset()

    def add(self, gt, predict):
        #per whole sequence
        super(OcrReport, self).add(gt, predict)
        #per character
        gt_len = len(gt)
        self.char_report_dict['total_per_char']  += gt_len

        edit_distance = self.levenshteinDistance(gt, predict)
        correct_len =  gt_len - edit_distance
        if correct_len > 0:
            self.char_report_dict['correct_per_char'] += correct_len

    def initReportFormat(self):
        super(OcrReport, self).initReportFormat()
        self.char_report_str = '''
|{}|{}|{}|{}|
|--|--|--|--|
|{}|{}|{}|{}|
        '''

    def initReportDict(self):
        super(OcrReport, self).initReportDict()
        self.char_report_dict = OrderedDict()
        self.char_report_dict['dataset'] = self.ds_name
        self.char_report_dict['total_per_char'] = 0
        self.char_report_dict['correct_per_char'] = 0
        self.char_report_dict['accuracy_per_char'] = 0

    #edit distance
    def levenshteinDistance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def __call__(self):
        super(OcrReport, self).__call__()
        assert self.char_report_dict['total_per_char'] > 0, 'total characters must grater than 0'
        self.char_report_dict['accuracy_per_char'] = self.char_report_dict['correct_per_char'] / self.char_report_dict['total_per_char']
        report_keys = [k for k,v in self.char_report_dict.items()]
        report_values = [v for k,v in self.char_report_dict.items() ]
        print(self.char_report_str.format(*report_keys, *report_values))

if __name__ == "__main__":
    print("==========test FaceReport============")
    report = FaceReport('gemfield',5)
    report.add("","2").add("gemfield","gemfield").add(1,1).add(None,None).add("1","1")
    report()

    print("==========test OcrReport============")
    report = OcrReport('gemfield',4)
    report.add('朝辞白帝彩云间', '朝辞白彩云间')
    report.add('君不见黄河之水天上来', '君不见黄河之水天上来')
    report.add('非汝之为美，美人之贻', '非汝之为美，美人之遗')
    report.add('gemfield', 'gem fie,ld')
    report()
