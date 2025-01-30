from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np



class HardConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(HardConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08

    def forward(self, features_1, features_2, pairsimi):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]

        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        all_sim = torch.mm(features, features.t().contiguous())
        neg = torch.exp(all_sim / self.temperature).masked_select(mask).view(2*batch_size, -1)

        pairmask = torch.cat([pairsimi, pairsimi], dim=0)
        posmask = (pairmask == 1).detach()
        posmask = posmask.type(torch.int32)

        negimp = neg.log().exp()
        Ng = (negimp*neg).sum(dim = -1) / negimp.mean(dim = -1)
        loss_pos = (-posmask * torch.log(pos / (Ng+pos))).sum() / posmask.sum()
        losses["instdisc_loss"] = loss_pos
        return losses

class iMIXConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(iMIXConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, features_1, features_2, mix_rand_list, mix_lambda):
        losses = {}

        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]
    
        all_sim = torch.mm(features_1, features_2.t().contiguous())/self.temperature
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos_rand = torch.exp(torch.sum(features_1*features_2[mix_rand_list], dim=-1) / self.temperature)
        
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = ~mask
        neg = torch.exp(all_sim).masked_select(mask).view(batch_size, -1) 
        negimp = neg.log().exp()
        Ng = (negimp*neg).sum(dim = -1) / negimp.mean(dim = -1)
        
        output = torch.log(pos / (Ng+pos))
        output_rand = torch.log(pos_rand / (Ng+pos))
        loss_pos = -(mix_lambda * output + (1. - mix_lambda) * output_rand).mean()
        
        losses["instdisc_loss"] = loss_pos
        return losses
    
class AttentionLoss(nn.Module):
    def __init__(self, distribution_weight=1.0, correlation_weight=1.0):
        super().__init__()
        self.distribution_weight = distribution_weight
        self.correlation_weight = correlation_weight
    
    def forward(self, attention_weights_1, attention_weights_2):
        """
        Args:
            attention_weights_1: [batch_size, seq_len_1, 1] - attention for first part of sequence
            attention_weights_2: [batch_size, seq_len_2, 1] - attention for second part of sequence
            embeddings_1: [batch_size, hidden_size] - weighted embeddings from first part
            embeddings_2: [batch_size, hidden_size] - weighted embeddings from second part
        """
        # Squeeze out the last dimension of attention weights
        attention_1 = attention_weights_1.squeeze(-1)
        attention_2 = attention_weights_2.squeeze(-1)
        
        # 1. Distribution matching loss
        # Compare statistical properties of attention distributions
        mean_1 = torch.mean(attention_1, dim=1)
        mean_2 = torch.mean(attention_2, dim=1)
        var_1 = torch.var(attention_1, dim=1)
        var_2 = torch.var(attention_2, dim=1)
        
        distribution_loss = (
            F.mse_loss(mean_1, mean_2) +  # Match means
            F.mse_loss(var_1, var_2)      # Match variances
        )
        
        # 2. Feature correlation loss
        # Encourage similar patterns of importance
        def get_top_k_pattern(attention, k=5):
            # Get binary pattern of top-k attention weights
            k = min(k, attention.size(1))
            _, top_k_indices = torch.topk(attention, k, dim=1)
            pattern = torch.zeros_like(attention)
            pattern.scatter_(1, top_k_indices, 1.0)
            return pattern
        
        pattern_1 = get_top_k_pattern(attention_1)
        pattern_2 = get_top_k_pattern(attention_2)
        
        # Compare distributions of attention patterns
        pattern_dist_1 = torch.mean(pattern_1, dim=0)
        pattern_dist_2 = torch.mean(pattern_2, dim=0)
        correlation_loss = F.kl_div(
            F.log_softmax(pattern_dist_1, dim=0),
            F.softmax(pattern_dist_2, dim=0),
            reduction='batchmean'
        )
        
        # Combine losses
        total_loss = (
            self.distribution_weight * distribution_loss +
            self.correlation_weight * correlation_loss
        )
        
        return total_loss, {
            'distribution_loss': distribution_loss.item(),
            'correlation_loss': correlation_loss.item()
        }