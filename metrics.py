# write distance metric here

import numpy as np


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

        
class AccumulatedDistanceMetric(Metric):
    """
        Accumates distance for an average to be calculated.
    """

    def __init__(self, metric_name):
        self.correct = 0
        self.total = 0
        self.metric_name = metric_name

    def __call__(self, outputs, target, loss):
        "We want to add the distance calculations here."
        pred = outputs.data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        self.total += target.size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return self.metric_name

