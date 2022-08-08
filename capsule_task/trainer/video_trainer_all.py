import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class VideoTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, use_amp,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):

        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.use_amp = use_amp
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.time_metrics = MetricTracker('time', 'batch_time', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target, paths)  in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):

                output, _ = self.model(data)
                #inputs, targets_a, targets_b, lam = mixup_data(data, target, 1, True)
                #output, _ = self.model(inputs)

                # reshape output and target for cross entropy loss
                output = output.reshape(output.size(0) * output.size(1), -1)  # (batch * seq_len x classes)
                target = target.reshape(-1)  # (batch * seq_len), class index
                #targets_a = targets_a.reshape(-1)  # (batch * seq_len), class index
                #targets_b = targets_b.reshape(-1)
                #loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)


                
                loss = self.criterion(output, target)

            if not torch.isfinite(loss):
                print('WARNING: non_finite loss, ending training')
                exit(1)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                #self.optimizer.step()
                output, _ = self.model(data)
                output = output.reshape(output.size(0) * output.size(1), -1)  # (batch * seq_len x classes)
                target = target.reshape(-1)  # (batch * seq_len), class index
                #targets_a = targets_a.reshape(-1)  # (batch * seq_len), class index
                #targets_b = targets_b.reshape(-1)
                self.criterion(output, target).backward()
                #mixup_criterion(self.criterion, output, targets_a, targets_b, lam).backward()
                self.optimizer.second_step(zero_grad=True)


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                
                self.train_metrics.update(met.__name__, met(output, target))
                #self.train_metrics.update(met.__name__, met(output, targets_a, targets_b, lam ))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))


            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, paths) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                
                #inputs, targets_a, targets_b, lam = mixup_data(data, target, 1, True)
                #output, _ = self.model(inputs)
        
                # reshape output and target for cross entropy loss
                output = output.reshape(output.size(0) * output.size(1), -1)  # (batch * seq_len x classes)
                target = target.reshape(-1)
                #targets_a = targets_a.reshape(-1)  # (batch * seq_len), class index
                #targets_b = targets_b.reshape(-1)
                #loss = mixup_criterion(self.criterion, output, targets_a, targets_b, lam)
                
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                    #self.valid_metrics.update(met.__name__, met(output, targets_a, targets_b, lam))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)