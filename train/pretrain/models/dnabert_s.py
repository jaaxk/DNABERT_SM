import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from DNABERT2_MIX.bert_layers import BertModel

class DNABert_S_Attention(nn.Module):
    def __init__(self, feat_dim=128, mix=True, model_mix_dict=None, load_dict=None, curriculum=False):
        super(DNABert_S_Attention, self).__init__()
        print("-----Initializing DNABert_S with Discriminative Attention-----")
        if (not mix) & (not curriculum):
            self.dnabert2 = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        else:
            self.dnabert2 = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.emb_size = self.dnabert2.pooler.dense.out_features
        self.feat_dim = feat_dim
        self.attn_hidden_size = 256

        # Discriminative attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.emb_size, self.attn_hidden_size),
            nn.Tanh(),
            nn.Linear(self.attn_hidden_size, 1)
        )

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))

        if load_dict is not None:
            self.dnabert2.load_state_dict(torch.load(load_dict+'pytorch_model.bin'))
            self.contrast_head.load_state_dict(torch.load(load_dict+'head_weights.ckpt'))
            # Load attention weights if they exist
            try:
                self.attention.load_state_dict(torch.load(load_dict+'attention_weights.ckpt'))
            except:
                print("No pretrained attention weights found. Starting with random initialization.")

    def compute_attention_weights(self, hidden_states, attention_mask):
        print("[DEBUG] Inside compute_attention_weights")
        print("  attention_mask shape:", attention_mask.shape)
        print("  attention_mask values:", attention_mask)
        
        attention_scores = self.attention(hidden_states).squeeze(-1)
        print("  attention_scores shape:", attention_scores.shape)
        
        attention_mask = attention_mask.bool()
        attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        print("  attention_weights shape:", attention_weights.shape)
        return attention_weights
    
    def forward(self, input_ids, attention_mask, task_type='train', mix=True, mix_alpha=1.0, mix_layer_num=-1):
        print("[DEBUG] Initial attention_mask shape:", attention_mask.shape)
        print("[DEBUG] Initial attention_mask values:", attention_mask)
        if task_type == "evaluate":
            return self.get_attention_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            if attention_mask is not None:
                print("[DEBUG] Before unbind - attention_mask shape:", attention_mask.shape)
                attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)
                print("[DEBUG] After unbind - attention_mask_1 shape:", attention_mask_1.shape)
                print("[DEBUG] After unbind - attention_mask_2 shape:", attention_mask_2.shape)
            
            print("Initial attention_mask_1 shape:", attention_mask_1.shape)
            print("Initial attention_mask_1 values:", torch.sum(attention_mask_1))

            if mix:
                bert_output_1, mix_rand_list, mix_lambda, attention_mask_1 = self.dnabert2.forward(
                    input_ids=input_ids_1, attention_mask=attention_mask_1, 
                    mix=mix, mix_alpha=mix_alpha, mix_layer_num=mix_layer_num
                )
                print("[DEBUG] After BertModel.forward - attention_mask_1 shape:", attention_mask_1.shape)
                print("[DEBUG] After BertModel.forward - attention_mask_1 values:", attention_mask_1)
                bert_output_2 = self.dnabert2.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            else:
                bert_output_1 = self.dnabert2.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
                bert_output_2 = self.dnabert2.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            #debugging  
            print("After BERT:")
            print("bert_output_1[0] shape:", bert_output_1[0].shape)
            print("attention_mask_1 shape:", attention_mask_1.shape)
            print("attention_mask_1 values:", torch.sum(attention_mask_1))   

            # Compute attention weights
            attention_weights_1 = self.compute_attention_weights(bert_output_1[0], attention_mask_1)
            if attention_weights_1 is not None:
                print("attention_weights_1 shape:", attention_weights_1.shape)
                print("attention_weights_1 sum:", torch.sum(attention_weights_1))
                print("attention_weights_1 min/max:", torch.min(attention_weights_1).item(), torch.max(attention_weights_1).item())
            else:
                print('attention_weights_1 is None')

            attention_weights_2 = self.compute_attention_weights(bert_output_2[0], attention_mask_2)

            # Apply attention weights
            weighted_output_1 = torch.sum(bert_output_1[0] * attention_weights_1.unsqueeze(-1), dim=1)
            weighted_output_2 = torch.sum(bert_output_2[0] * attention_weights_2.unsqueeze(-1), dim=1)

            cnst_feat1, cnst_feat2 = self.contrast_logits(weighted_output_1, weighted_output_2) #Since cnst_feat contain the contrastive learning logits for the weighted outputs, attention weights dont have to be passed to loss function

            if mix:
                return cnst_feat1, cnst_feat2, attention_weights_1, attention_weights_2, mix_rand_list, mix_lambda, weighted_output_1, weighted_output_2
            else:
                return cnst_feat1, cnst_feat2, weighted_output_1, weighted_output_2

    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    def get_attention_embeddings(self, input_ids, attention_mask):
        bert_output = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)
        attention_weights = self.compute_attention_weights(bert_output[0], attention_mask)
        embeddings = torch.sum(bert_output[0] * attention_weights, dim=1)
        return embeddings, attention_weights  # Return weights for analysis if needed
