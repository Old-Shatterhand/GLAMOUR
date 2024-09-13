import torch
import numpy as np
import torchmetrics


class Meter_v3:
    def __init__(self, mean=None, std=None):
        '''
        Initializes a Meter_v2 object

        Args:
        mean : torch.float32 tensor of shape (T) or None, mean of existing training labels across tasks
        std : torch.float32 tensor of shape (T) or None, std of existing training labels across tasks

        '''
        self._mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self._mean = mean.cpu()
            self._std = std.cpu()
        else:
            self._mean = None
            self._std = None

    def update(self, y_pred, y_true, mask=None):
        '''Updates for the result of an iteration

        Args:
        y_pred : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        y_true : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        mask : None or float32 tensor, binary mask indicating the existence of ground truth labels
        '''
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self._mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self._mask.append(mask.detach().cpu())

    def _finalize(self):
        '''Utility function for preparing for evaluation.

        Returns:
        mask : float32 tensor, binary mask indicating the existence of ground truth labels
        y_pred : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        y_true : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        '''
        mask = torch.cat(self._mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self._mean is not None) and (self._std is not None):
            y_pred = y_pred * self._std + self._mean

        return mask, y_pred, y_true

    def compute_metric(self, metric_name, reduction='none'):
        '''Computes the metric of interest

        Args:
        metric_name : str, name of the metric to compute
        reduction : str, reduction method for the metric

        Returns:
        metric : float, computed metric
        '''
        mask, y_pred, y_true = self._finalize()
        if metric_name == 'mae':
            return torchmetrics.functional.mean_absolute_error(y_true, y_pred)
        elif metric_name == 'rmse':
            return torchmetrics.functional.mean_squared_error(y_true, y_pred).sqrt()
        elif metric_name == 'mse':
            return torchmetrics.functional.mean_squared_error(y_true, y_pred)
        elif metric_name == 'r2':
            return torchmetrics.functional.r2_score(y_true, y_pred)
        elif metric_name == 'auroc':
            return torchmetrics.functional.auroc(y_pred, y_true)
        elif metric_name == 'matthews_corrcoef':
            return torchmetrics.functional.matthews_corrcoef(y_pred, y_true)
        elif metric_name == 'accuracy':
            return torchmetrics.functional.accuracy(y_pred, y_true)
        else:
            raise ValueError(f'Unsupported metric: {metric_name}')
