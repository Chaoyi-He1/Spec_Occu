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
    metric_logger.add_meter('acc_steps', SmoothedValue(window_size=1, fmt='{value:.6f}'))
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
                                            ret_traj=False, sampling="ddpm") / 50
            
            predict_labels = [T2F_model(predict) for predict in predicts]
            acc_steps, acc, F1score, predict_probs = calculate_prob_cloud(predict_labels, future_labels)
            BCELoss, _ = zip(*[T2F_criterion(predict_label, future_labels) for predict_label in predict_labels])
            
        
        if F1score.mean() > best_F1score and acc > best_acc:
            best_history, best_hist_labels, best_future, best_future_labels = history, hist_labels, future, future_labels
            best_acc, best_F1score = acc, F1score.mean()
            
        metric_logger.update(loss=torch.stack(BCELoss).mean().item(), 
                             acc=acc.item(), acc_steps=acc_steps.item(),
                             F1_score=F1score.mean().item())
    
    return best_history, best_hist_labels, best_future, best_future_labels, \
           {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def calculate_prob_cloud(predict_labels: List[Tensor], future_labels: Tensor, 
                         threshold: float = 0.5):
    predict_labels = torch.sigmoid(torch.stack(predict_labels)).round()
    sample_size, num_time_steps, num_classes = predict_labels.shape
    predict_probs = predict_labels.sum(dim=0) / sample_size
    predict_labels = predict_probs.clone()
    predict_labels[predict_probs < threshold] = 0
    predict_labels[predict_probs >= threshold] = 1
    
    # calculate the accuracy and F1 score
    acc_steps = torch.eq(predict_labels, future_labels).sum(dim=1).float() / num_classes
    acc = acc_steps.mean()
    tp = torch.logical_and(predict_labels == 1, future_labels == 1).sum(dim=1).float()
    fp = torch.logical_and(predict_labels == 1, future_labels == 0).sum(dim=1).float()
    fn = torch.logical_and(predict_labels == 0, future_labels == 1).sum(dim=1).float()
    F1scores = tp / (tp + 0.5 * (fp + fn))
    return acc_steps, acc, F1scores, predict_probs
    