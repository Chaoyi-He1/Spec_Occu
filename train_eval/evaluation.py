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
from eval_visual.visualization import *
from eval_visual.visual_CPC import *


def evaluate(encoder: torch.nn.Module, diff_model: torch.nn.Module, 
             T2F_model: torch.nn.Module, diff_criterion: Diffusion_utils, 
             T2F_criterion: Temporal_Freq_Loss, data_loader: Iterable, 
             device: torch.device, scaler=None, is_train=False):
    encoder.eval()
    diff_model.eval()
    T2F_model.eval()
    diff_criterion.eval()
    T2F_criterion.eval()
    metric_logger = MetricLogger(delimiter="; ")
    # metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('acc_steps', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('F1_score', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Test:'
    
    all_predictions, all_true_labels = [], []
    all_history, all_hist_labels = [], []
    
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
            
            predict_labels = torch.stack([T2F_model(predict) for predict in predicts], dim=0)  
            
            all_predictions.append(predict_labels.detach().cpu())
            all_true_labels.append(future_labels.detach().cpu())
            all_history.append(history.detach().cpu())
            all_hist_labels.append(hist_labels.detach().cpu())
            
    all_predictions = torch.cat(all_predictions, dim=1)
    all_true_labels = torch.cat(all_true_labels, dim=0)
    all_history = torch.cat(all_history, dim=0)
    all_hist_labels = torch.cat(all_hist_labels, dim=0)
    best_index = calculate_prob_cloud(all_predictions, all_true_labels, 
                                      history=all_history, hist_labels=all_hist_labels,
                                      is_train=is_train)
    return best_index, all_predictions[:, best_index, ...], all_true_labels[best_index, ...]
           


def calculate_prob_cloud(predicts: Tensor, future_labels: Tensor, 
                         history: Tensor, hist_labels: Tensor,
                         is_train=False):
    # predict_labels (num_samples, batch_size, num_time_steps, num_classes)
    # Sweep the threshold from 0 to 80 with 100 steps to find the best threshold for each prediction in the batch
    # the best threshold is the one that at the left top corner of the ROC curve (diagonal line of the ROC curve)
    # predicts = torch.stack(predicts, dim=0)
    best_thresholds = np.zeros(predicts.shape[1])
    best_rate = torch.zeros(predicts.shape[1])
    best_acc = torch.zeros(predicts.shape[1])
    
    for threshold in np.linspace(10, 80, 100):
        prediction_labels = torch.round((predicts > threshold).float().mean(dim=0))
        
        # find if the prediction at this threshold is the best in the ROC curve for each prediction in the batch
        TP = (prediction_labels * future_labels).sum(dim=(1, 2))
        TN = ((1 - prediction_labels) * (1 - future_labels)).sum(dim=(1, 2))
        FP = (prediction_labels * (1 - future_labels)).sum(dim=(1, 2))
        FN = ((1 - prediction_labels) * future_labels).sum(dim=(1, 2))

        # calculate the TPR and FPR for each prediction in the batch at this threshold
        # and find if it is at the diagonal line of the ROC curve from the left top corner to the right bottom corner
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        
        # chech if the TPR and FPR make the prediction at the left top corner of the ROC curve
        better_ratio_mask = (TPR >= -FPR + 1) & (TPR / FPR > best_rate)
        best_rate[better_ratio_mask] = (TPR / FPR)[better_ratio_mask]
        best_thresholds[better_ratio_mask] = threshold
        best_acc[better_ratio_mask] = ((TP + TN) / (TP + TN + FP + FN))[better_ratio_mask]
    
    # find the top 10 best in the batch
    best_10_index = torch.argsort(best_rate, descending=True)[:10].cpu().numpy()
    # randomly select 10 from the top 100 best
    # best_10_index = np.random.choice(best_100_index, 10)
    for i, best_index in enumerate(best_10_index):
        #generate the probability cloud for the best prediction in the batch
        best_pred = predicts[:, best_index, :, :]
        
        best_threshold = best_thresholds[best_index]
        best_prob_cloud = (predicts[:, best_index, :, :] > best_threshold).float().mean(dim=0)
        best_prob_cloud = best_prob_cloud
        
        best_true_labels = future_labels[best_index, :, :]
        
        best_history = history[best_index, :, :, :]
        best_hist_labels = hist_labels[best_index, :, :]
        
        #save the probability cloud matrix and the true labels matrix
        if is_train:
            np.savetxt(f"prob_cloud/train_best_prob_cloud_{i}.csv", best_prob_cloud.cpu().numpy(), delimiter=",")
            np.savetxt(f"prob_cloud/train_best_true_labels_{i}.csv", best_true_labels.cpu().numpy(), delimiter=",")
            # np.savetxt(f"prob_cloud/train_best_history_{i}.csv", best_history.cpu().numpy(), delimiter=",")
            np.savetxt(f"prob_cloud/train_best_hist_labels_{i}.csv", best_hist_labels.cpu().numpy(), delimiter=",")
            
            if not os.path.exists(f"prob_cloud/train_best_pred_{i}"):
                os.makedirs(f"prob_cloud/train_best_pred_{i}")
            np.savetxt(f"prob_cloud/train_best_pred_{i}/train_best_threshold_{i}.csv", np.array([best_threshold]), delimiter=",")
            for j in range(20):
                np.savetxt(f"prob_cloud/train_best_pred_{i}/train_best_pred_{i}_{j}.csv", best_pred[j].cpu().numpy(), delimiter=",")
        else:
            np.savetxt(f"prob_cloud/test_best_prob_cloud_{i}.csv", best_prob_cloud.cpu().numpy(), delimiter=",")
            np.savetxt(f"prob_cloud/test_best_true_labels_{i}.csv", best_true_labels.cpu().numpy(), delimiter=",")
            # np.savetxt(f"prob_cloud/test_best_history_{i}.csv", best_history.cpu().numpy(), delimiter=",")
            np.savetxt(f"prob_cloud/test_best_hist_labels_{i}.csv", best_hist_labels.cpu().numpy(), delimiter=",")
            
            if not os.path.exists(f"prob_cloud/test_best_pred_{i}"):
                os.makedirs(f"prob_cloud/test_best_pred_{i}")
            np.savetxt(f"prob_cloud/test_best_pred_{i}/test_best_threshold_{i}.csv", np.array([best_threshold]), delimiter=",")
            for j in range(20):
                np.savetxt(f"prob_cloud/test_best_pred_{i}/test_best_pred_{i}_{j}.csv", best_pred[j].cpu().numpy(), delimiter=",")
    return best_10_index
    

def CPC_test(encoder: torch.nn.Module, diff_model: torch.nn.Module, 
             T2F_model: torch.nn.Module, diff_criterion: Diffusion_utils, 
             T2F_criterion: Temporal_Freq_Loss, data_loader: Iterable, 
             device: torch.device, scaler=None):
    encoder.eval()
    diff_model.eval()
    T2F_model.eval()
    diff_criterion.eval()
    T2F_criterion.eval()
    metric_logger = MetricLogger(delimiter="; ")
    all_features, all_labels = [], []
    header = 'Test:'
    
    for _, (history, _, _, _, index) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        torch.autograd.set_detect_anomaly(True)
        history = history.to(device)
        
        # Compute the output
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=scaler is not None):
            features = encoder(history)
            all_features.append(features.detach().cpu().numpy())
            all_labels.append(index.detach().cpu().numpy())
            
    print("Averaged stats:", metric_logger)
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    t_SNE_CPC(all_features, all_labels)
    return all_features, all_labels
