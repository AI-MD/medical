import torch

class LearningRateWarmup(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None) -> None:
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_schduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr * float(cur_iteration) / float(self.warmup_iteration)
        for param_group in self.optimizer.param_group:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_schduler.step(cur_iteration - self.warmup_iteration)
