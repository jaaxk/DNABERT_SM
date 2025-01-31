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

    def compute_attention_weights(self, sequence_output, attention_mask):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            attention_weights: [batch_size, seq_len, 1]
        """
        if attention_mask is None:
            print('attention_mask is None')
            return

        # Add dimension checks
        batch_size, seq_len, hidden_size = sequence_output.size()
        assert attention_mask.size() == (batch_size, seq_len), f"Attention mask shape mismatch. Expected {(batch_size, seq_len)}, got {attention_mask.size()}"
        
        # Generate attention scores
        attention_weights = self.attention(sequence_output)  # [batch_size, seq_len, 1]
        
        # Mask out padding tokens
        attention_mask = attention_mask.bool().unsqueeze(-1)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.masked_fill(~attention_mask, float('-inf'))
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Add shape check before returning
        assert attention_weights.size() == (batch_size, seq_len, 1), \
            f"Output shape mismatch. Expected {(batch_size, seq_len, 1)}, got {attention_weights.size()}"
        
        return attention_weights

    def forward(self, input_ids, attention_mask, task_type='train', mix=True, mix_alpha=1.0, mix_layer_num=-1):
        if task_type == "evaluate":
            return self.get_attention_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)

            if mix:
                bert_output_1, mix_rand_list, mix_lambda, attention_mask_1 = self.dnabert2.forward(
                    input_ids=input_ids_1, attention_mask=attention_mask_1, 
                    mix=mix, mix_alpha=mix_alpha, mix_layer_num=mix_layer_num
                )
                bert_output_2 = self.dnabert2.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            else:
                bert_output_1 = self.dnabert2.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
                bert_output_2 = self.dnabert2.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            #debugging  
            print(f'bert_output_1[0].size() = {bert_output_1[0].size()}')
            print(f'attention_mask_1.size() = {attention_mask_1.size()}')   

            # Compute attention weights
            attention_weights_1 = self.compute_attention_weights(bert_output_1[0], attention_mask_1)
            attention_weights_2 = self.compute_attention_weights(bert_output_2[0], attention_mask_2)

            # Apply attention weights
            weighted_output_1 = torch.sum(bert_output_1[0] * attention_weights_1, dim=1)
            weighted_output_2 = torch.sum(bert_output_2[0] * attention_weights_2, dim=1)

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
