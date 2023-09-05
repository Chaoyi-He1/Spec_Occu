import torch
import torch.nn as nn
from util.misc import *
from util.diffusion import *
from typing import Iterable
from itertools import chain
from util.Temp_to_Freq import F1_score
import torch.nn.functional as F
from util.diffusion import Diffusion_utils
from util.Temp_to_Freq import Temporal_Freq_Loss, F1_score


def evaluate(encoder: torch.nn.Module, diff_model: torch.nn.Module, 
                    T2F_model: torch.nn.Module, diff_criterion: Diffusion_utils, 
                    T2F_criterion: Temporal_Freq_Loss, data_loader: Iterable, 
                    device: torch.device, scaler=None):
    encoder.eval()
    diff_model.eval()
    T2F_model.eval()
    diff_criterion.eval()
    T2F_criterion.eval()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('F1_score', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'
    
    best_history, best_hist_labels, best_future, best_future_labels = None, None, None, None
    best_acc, best_F1score = 0, 0
    
    for _, (history, hist_labels,
            future, future_labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        torch.autograd.set_detect_anomaly(True)
        history = history.to(device)
        future = future.to(device)
        future_labels = future_labels.to(device)
        
        # Compute the output
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            features = encoder(history)
            predicts = diff_criterion.sample(num_points=future.shape[1], context=features,
                                            sample=20, bestof=False, step=10, flexibility=0.0, 
                                            model=diff_model, point_dim=future.shape[-1], 
                                            ret_traj=False, sampling="ddpm")
            
            predict_labels = [T2F_model(predict) for predict in predicts]
            BCELoss, acc_steps = [T2F_criterion(predict_label, future_labels) for predict_label in predict_labels]
            F1scores = [F1_score(predict_label, future_labels) for predict_label in predict_labels]
            acc = torch.stack(acc_steps).mean()
            F1score = torch.stack(F1scores).mean(dim=0)
        
        if F1score.mean() > best_F1score and acc > best_acc:
            best_history, best_hist_labels, best_future, best_future_labels = history, hist_labels, future, future_labels
            best_acc, best_F1score = acc, F1score.mean()
            
        metric_logger.update(loss=BCELoss.item(), acc=acc.item(), F1_score=F1score.mean().item())
    
    return best_history, best_hist_labels, best_future, best_future_labels, \
           {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    