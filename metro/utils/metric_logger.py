"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems and the Max Planck Institute for Biological
Cybernetics. All rights reserved.

Contact: ps-license@tuebingen.mpg.de

"""

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class EvalMetricsLogger(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        # define a upper-bound performance (worst case) 
        # numbers are in unit millimeter
        self.PAmPJPE = 100.0/1000.0
        self.mPJPE = 100.0/1000.0
        self.mPVE = 100.0/1000.0

        self.epoch = 0

    def update(self, mPVE, mPJPE, PAmPJPE, epoch):
        self.PAmPJPE = PAmPJPE
        self.mPJPE = mPJPE
        self.mPVE = mPVE
        self.epoch = epoch


class DetMetricsLogger(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        # define a worst case

        self.precision = .0
        self.recall = .0
        self.f1 = .0
        self.mPVE = 100.0/1000.0

        self.fp_error = 0
        self.fn_error = 0

        self.epoch = 0

    def update(self, mPVE, p, r, f1, fp_error, fn_error, epoch):
        self.precision = p
        self.recall = r
        self.f1 = f1

        self.fp_error = fp_error
        self.fn_error = fn_error
        
        self.mPVE = mPVE
        self.epoch = epoch
