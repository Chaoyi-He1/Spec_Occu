import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from util.misc import *
from util.diffusion import *
from typing import Iterable
from itertools import chain
from util.Temp_to_Freq import F1_score
import torch.nn.functional as F
from util.diffusion import Diffusion_utils
from util.Temp_to_Freq import Temporal_Freq_Loss, F1_score


def train_one_epoch(encoder: torch.nn.Module, diff_model: torch.nn.Module, 
                    T2F_model: torch.nn.Module, diff_criterion: Diffusion_utils, 
                    T2F_criterion: Temporal_Freq_Loss, data_loader: Iterable, 
                    diff_optimizer: torch.optim.Optimizer, T2F_optimizer: torch.optim.Optimizer,
                    ALL_optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0.01, 
                    scaler=None, freeze_encoder: bool =False):
    encoder.train()
    diff_model.train()
    T2F_model.train()
    diff_criterion.train()
    T2F_criterion.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('F1_score', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for _, (history, hist_labels,
            future, future_labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # torch.autograd.set_detect_anomaly(True)
        history = history.to(device)
        future = future.to(device)
        future_labels = future_labels.to(device)
            
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            features = encoder(history)
            BCELoss, _ = diff_criterion.get_loss_fine_tune(x_0=future, context=features, model=diff_model)
        
            loss = BCELoss.mean()
            # F1score = F1_score(predict_label, future_labels)

        if torch.isnan(loss):
            raise ValueError('NaN loss detected')

        # Backward
        diff_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if max_norm > 0:
            params = diff_model.parameters() if freeze_encoder \
                else chain(encoder.parameters(), diff_model.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None), torch.no_grad():
            features = encoder(history)
            predict = diff_criterion.sample(num_points=future.shape[1], context=features, 
                                            sample=1, bestof=False, step=10,
                                            model=diff_model, point_dim=future.shape[-1], 
                                            flexibility=0.0, ret_traj=False, sampling="ddpm")
            predict = predict[0].detach() # / 50.0
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            predict_label = T2F_model(predict)
            # print(torch.isnan(predict_label).any())
            loss_T2F, acc_steps = T2F_criterion(predict_label, future_labels)
            # print(loss_T2F.item())
            acc = acc_steps.mean()
            F1score = F1_score(predict_label, future_labels)
            
        if torch.isnan(loss_T2F):
            raise ValueError('NaN loss detected')
        loss += loss_T2F
            
        # reduce losses over all GPUs for logging purposes
        loss_reduced = reduce_loss(loss)
        acc_reduced = reduce_loss(acc)

        # Backward
        T2F_optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss_T2F).backward()
        else:
            loss_T2F.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(T2F_model.parameters(), max_norm)
            
        if scaler is not None:
            scaler.step(diff_optimizer)
            scaler.step(T2F_optimizer)
            scaler.update()
        else:
            diff_optimizer.step()
            T2F_optimizer.step()
        # torch.autograd.set_detect_anomaly(False)
        
        # Update metric
        metric_logger.update(loss=loss_reduced.item())
        metric_logger.update(lr=diff_optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=acc_reduced.item())
        metric_logger.update(F1_score=torch.stack(F1score).mean().item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(encoder: torch.nn.Module, diff_model: torch.nn.Module, 
             T2F_model: torch.nn.Module, diff_criterion: Diffusion_utils, 
             T2F_criterion: Temporal_Freq_Loss, data_loader: Iterable, 
             device: torch.device, scaler=None, repeat=20, epoch=0):
    
    encoder.eval()
    diff_model.eval()
    T2F_model.eval()
    diff_criterion.eval()
    T2F_criterion.eval()
    
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('acc', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('F1_score', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('ADE_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('FDE_loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ADE_percentage', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('FDE_percentage', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'

    all_predictions = []
    all_true_labels = []
    for _, (history, hist_labels,
            future, future_labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        history = history.to(device)
        future = future.to(device)
        future_labels = future_labels.to(device)
        b, l, c, d = future.shape

        # Compute the output
        with torch.cuda.amp.autocast(enabled=scaler is not None), torch.no_grad():
            features = encoder(history)
            predicts = diff_criterion.sample(num_points=l, context=features, sample=repeat, bestof=False, step=10,
                                            model=diff_model, point_dim=d, flexibility=0.0, ret_traj=False, sampling="ddpm")
            predict_label = torch.stack([T2F_model(predict) for predict in predicts], dim=0)
            
            loss_T2F, acc_steps = T2F_criterion(predict_label.mean(dim=0), future_labels)
            acc = acc_steps.mean()
            F1score = F1_score(predict_label.mean(dim=0), future_labels)
            
            all_predictions.append(predict_label.detach().cpu())
            all_true_labels.append(future_labels.detach().cpu())
            
        ADE, FDE, ADE_percents, FDE_percents = compute_batch_statistics(predicts, future)
        
        # reduce losses over all GPUs for logging purposes
        acc_reduced = reduce_loss(acc)
        F1score_reduced = reduce_loss(torch.stack(F1score).mean())
        ADE_reduced = reduce_loss(ADE)
        FDE_reduced = reduce_loss(FDE)
        ADE_percents_reduced = reduce_loss(ADE_percents)
        FDE_percents_reduced = reduce_loss(FDE_percents)

        # Update metric
        metric_logger.update(acc=acc_reduced.item())
        metric_logger.update(F1_score=F1score_reduced.item())
        metric_logger.update(ADE_loss=ADE_reduced.mean().item())
        metric_logger.update(FDE_loss=FDE_reduced.mean().item())
        metric_logger.update(ADE_percentage=ADE_percents_reduced.mean().item())
        metric_logger.update(FDE_percentage=FDE_percents_reduced.mean().item())
    
    # Transfer the all_predictions from list to tensor and compute the precision and recall for different thresholds, plot the ROC curve
    # Concat the all_predictions among the batch dimension
    all_predictions = torch.cat(all_predictions, dim=1).numpy()
    all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
    # Compute the TPR and FPR for different thresholds
    TPRs = []
    FPRs = []
    for threshold in np.linspace(10, 80, 500):
        all_predictions_label = (all_predictions >= threshold).astype(int)
        all_predictions_label = np.round(all_predictions_label.sum(axis=0) / repeat)
        
        # Compute the precision and recall
        TP = (all_predictions_label * all_true_labels).sum()
        TN = ((1 - all_predictions_label) * (1 - all_true_labels)).sum()
        FP = (all_predictions_label * (1 - all_true_labels)).sum()
        FN = ((1 - all_predictions_label) * all_true_labels).sum()
        
        assert (TP + FN) != 0 and (FP + TN) != 0, "TP + FN and FP + TN should not be zero"
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        
        
        TPRs.append(TPR)
        FPRs.append(FPR)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # Plot the ROC curve, write each threshold's value near each point
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(FPRs, TPRs, label="ROC curve for the model")
    # for i, txt in enumerate(np.linspace(10, 80, 100)):
    #     ax.annotate("{:.2f}".format(txt), (FPRs[i] + 0.1, TPRs[i] - 0.1), fontsize=8)
    ax.set_xlabel("FP Rate")
    ax.set_ylabel("TP Rate")
    ax.set_title("ROC curve")
    plt.legend()
    plt.grid(True)
    # if fine_tune_roc folder does not exist, create it
    if not os.path.exists("fine_tune_roc"):
        os.makedirs("fine_tune_roc")
    plt.savefig("fine_tune_roc/ROC_curve_Epoch_{}.png".format(epoch))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
