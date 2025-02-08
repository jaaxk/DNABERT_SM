import os
import sys
import csv
import json
import numpy as np
import random
import torch
import torch.nn as nn
from textaugment import EDA
from tqdm import tqdm
from utils.contrastive_utils import HardConLoss, iMIXConLoss
import torch.distributed as dist

class Trainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, val_loader, args, rank):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gstep = 0
        self.start_epoch = 0
        self.rank = rank

        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        self.attn_loss_weight = 1 #Can be tuned!!!!
        if args.con_method == 'mutate':
            self.data_mutate = EDA()
        self.hard_loss = HardConLoss(temperature=self.args.temperature).cuda()
        self.imix_loss = iMIXConLoss(temperature=self.args.temperature).cuda()
        self.curriculum = args.curriculum

    def get_batch_token(self, dna_seq):
        max_length = self.args.max_length
        token_feat = self.tokenizer.batch_encode_plus(
            dna_seq, 
            max_length=max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat

    def dna_complement(self, dna_seq):
        # Define the mapping of each nucleotide to its complement
        complement = {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G': 'C'
        }
        complemented_sequence = ''.join([complement[nucleotide] for nucleotide in dna_seq])
        return complemented_sequence

    def dna_swap(self, dna_seq, data_aug, num):
        for i in range(num):
            swap_sequence = data_aug.random_swap(dna_seq)
        return swap_sequence

    def dna_delete(self, dna_seq, data_aug, p=0.05):
        dna_seq = ' '.join(dna_seq)
        delete_sequence = data_aug.random_deletion(dna_seq,p)
        delete_sequence = delete_sequence.replace(' ','')
        return delete_sequence
        
    def prepare_pairwise_input(self, batch):
        text1, text2, pairsimi = batch['seq1'], batch['seq2'], batch['pairsimi'].cuda()
        # Tokenize the feature2, depending on different data augmentation method
        # including "same_species", "dropout", "double_strand", "mutate"
        feat1 = self.get_batch_token(text1)
        if self.args.con_method=="same_species" or self.args.con_method=="dropout":
            feat2 = self.get_batch_token(text2)
        elif self.args.con_method=="double_strand":
            text2_complement = []
            for i in range(len(text2)):
                text2_complement.append(self.dna_complement(text2[i]))
            feat2 = self.get_batch_token(text2_complement)
        elif self.args.con_method=="mutate":
            text2_swap = []
            for i in range(len(text2)):
                text2_swap.append(self.dna_swap(text2[i], self.data_mutate, num=int(0.05*len(text2[i]))))
            text2_delete = []
            for i in range(len(text2_swap)):
                text2_delete.append(self.dna_delete(text2_swap[i], self.data_mutate, p=0.05))
            feat2 = self.get_batch_token(text2_delete)

        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()
    
    """ def broadcast_state_dict(self, state_dict):
        # Convert state dict to list of tensors for broadcasting
        tensor_names = list(state_dict.keys())
        tensor_values = [state_dict[name] for name in tensor_names]
        
        # Broadcast tensors
        for i in range(len(tensor_values)):
            tensor_values[i] = tensor_values[i].to(self.device)
            dist.broadcast(tensor_values[i], src=0)
        
        # Recreate state dict
        return {name: value for name, value in zip(tensor_names, tensor_values)} """

    def load_state_dicts(self, load_dir, load_optimizer=False):
        # Only rank 0 needs to load the files
        if self.rank == 0:
            model_state = torch.load(load_dir + '/pytorch_model.bin')
            contrast_state = torch.load(load_dir + '/con_weights.ckpt')
            attention_state = torch.load(load_dir + '/attention_weights.ckpt')
            if load_optimizer:
                optimizer_state = torch.load(load_dir + '/checkpoint.pt')['optimizer_state_dict']
        
            # Load the state dicts into the model
            self.model.module.dnabert2.load_state_dict(model_state)
            self.model.module.contrast_head.load_state_dict(contrast_state)
            self.model.module.attention.load_state_dict(attention_state)
            if load_optimizer:
                self.optimizer.load_state_dict(optimizer_state)

        # Make sure all processes sync up
        dist.barrier()


    def save_model(self, step=None, epoch=None, save_best=False):
        if self.rank == 0:
            if save_best:
                save_dir = os.path.join(self.args.resPath, 'best')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.model.module.dnabert2.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.model.module.contrast_head.state_dict(), save_dir+"/con_weights.ckpt")
                torch.save(self.model.module.attention.state_dict(), save_dir+"/attention_weights.ckpt")
                # Modify config file
                if self.args.mix:
                    config_file_path = save_dir+"/config.json"
                    with open(config_file_path, "r") as file:
                        config_data = json.load(file)
                    base_path = config_data["_name_or_path"]
                    for key in config_data['auto_map']:
                        config_data['auto_map'][key] = f"{base_path}--{config_data['auto_map'][key]}"
                    with open(config_file_path, 'w') as file:
                        json.dump(config_data, file, indent=4)
            else:
                save_dir = os.path.join(self.args.resPath, str(step))
                self.last_saved_step = step
                last_saved_step_tensor = torch.tensor([int(step)], dtype=torch.long).to(self.device)
                dist.broadcast(last_saved_step_tensor, src=0)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.model.module.dnabert2.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.model.module.contrast_head.state_dict(), save_dir+"/con_weights.ckpt")
                torch.save(self.model.module.attention.state_dict(), save_dir+"/attention_weights.ckpt")
                torch.save({
                    'epoch': epoch,
                    'gstep': self.gstep,
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, save_dir+'/checkpoint.pt')

                # Modify config file
                if self.args.mix:
                    config_file_path = save_dir+"/config.json"
                    with open(config_file_path, "r") as file:
                        config_data = json.load(file)
                    base_path = config_data["_name_or_path"]
                    for key in config_data['auto_map']:
                        config_data['auto_map'][key] = f"{base_path}--{config_data['auto_map'][key]}"
                    with open(config_file_path, 'w') as file:
                        json.dump(config_data, file, indent=4)

        else:
            if not save_best:
                last_saved_step_tensor = torch.tensor([0], dtype=torch.long).to(self.device)
                dist.broadcast(last_saved_step_tensor, src=0)
                self.last_saved_step = str(last_saved_step_tensor.item())

    def load_checkpoint(self):
        if os.path.exists(self.args.resPath):
            print_once(f'{self.args.resPath} exists')
            dirs = [d for d in os.listdir(self.args.resPath) if os.path.isdir(os.path.join(self.args.resPath, d))]
            checkpoints = [int(d) for d in dirs if d.isdigit()]
            
            latest_checkpoint = os.path.join(self.args.resPath, str(max(checkpoints)))
            print_once(f'Loading from checkpoint {latest_checkpoint}')
            self.load_state_dicts(latest_checkpoint, load_optimizer=True)
            checkpoint_data = torch.load(latest_checkpoint + '/checkpoint.pt', map_location=self.device)
            self.start_epoch = checkpoint_data['epoch']
            self.gstep = checkpoint_data['gstep']
            print_once(f'Resumed from epoch {self.start_epoch}, step {self.gstep}')

        else:
            print_once('No checkpoint to resume from')
            return

    def check_attention_gradients(self):
        has_grad = False
        for param in self.model.module.attention.parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_grad = True
                break
        return has_grad

    def train_step(self, input_ids, attention_mask, pairsimi, curriculum_not_start=True):    
        with torch.autocast(device_type="cuda"):
            if (not self.args.mix) or (self.curriculum & curriculum_not_start):
                feat1, feat2, _, _ = self.model(input_ids, attention_mask, mix=False)
                losses = self.hard_loss(feat1, feat2, pairsimi)
                loss = losses["instdisc_loss"]

            else:
                if self.args.mix_layer_num != -1:
                    feat1, feat2, attn1, attn2, mix_rand_list, mix_lambda, _, _ = self.model(input_ids, attention_mask, \
                        mix=self.args.mix, mix_alpha=self.args.mix_alpha, mix_layer_num=self.args.mix_layer_num)
                else:
                    feat1, feat2, attn1, attn2, mix_rand_list, mix_lambda, _, _ = self.model(input_ids, attention_mask, \
                        mix=self.args.mix, mix_alpha=self.args.mix_alpha)
                
                losses = self.imix_loss(feat1, feat2, mix_rand_list, mix_lambda)
                loss = losses["instdisc_loss"] #why???
        
        loss.backward()
        #if self.rank == 0:
        #    print_once(f"Loss: {loss.item()}")

        """ if self.check_attention_gradients():
            print_once('Gradients successfully updating')
        else:
            print_once('Gradients NOT updating') """

        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    def train(self):   
        self.load_checkpoint()
        #Synchronize self.gstep across all ranks
        gstep_tensor = torch.tensor([self.gstep], dtype=torch.long).to(self.device)
        dist.all_reduce(gstep_tensor, op=dist.ReduceOp.MAX)
        self.gstep = gstep_tensor.item()
        self.all_iter = self.args.epochs * len(self.train_loader)
        print_once('\n={}/{}=Iterations/Batches'.format(self.all_iter, len(self.train_loader)))

        self.model.train()
        epoch_iterator = tqdm(self.train_loader, desc="Iteration") if self.rank == 0 else self.train_loader
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_loader.sampler.set_epoch(epoch) #shuffle data differently each epoch
            if self.curriculum:
                if self.args.epochs >=3:
                    if (epoch >= int(self.args.epochs/3)) & (epoch < int(self.args.epochs/3)+1):
                        load_dir = os.path.join(self.args.resPath, str(self.last_saved_step))
                        self.load_state_dicts(load_dir)
                        print_once('Curriculum learning: load model trained with stage I')
                    for j, batch in enumerate(epoch_iterator):
                        input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                        if epoch < int(self.args.epochs/3):
                            losses = self.train_step(input_ids, attention_mask, pairsimi)
                        else:
                            losses = self.train_step(input_ids, attention_mask, pairsimi, curriculum_not_start=False)
                        if self.gstep%self.args.logging_step==0:
                            self.save_model(step=self.gstep, epoch=epoch)
                        if self.gstep > self.args.logging_step*self.args.logging_num:
                            break
                        self.gstep += 1
                    
            else:
                for j, batch in enumerate(epoch_iterator):
                    input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    losses = self.train_step(input_ids, attention_mask, pairsimi)
                    if self.gstep%self.args.logging_step==0:
                        self.save_model(step=self.gstep, epoch=epoch)
                    if self.gstep > self.args.logging_step*self.args.logging_num:
                        break
                    self.gstep += 1
                
            print_once("Finish Epoch: ", epoch)
        return None
    
    def val(self):
        self.model.eval()
        best_checkpoint = 0
        best_val_loss = 10000
        for step in tqdm(range(self.args.logging_step, np.min([self.all_iter, self.args.logging_step*self.args.logging_num+1]), self.args.logging_step)):
            load_dir = os.path.join(self.args.resPath, str(step))
            self.load_state_dicts(load_dir)
            val_loss = 0.
            for j, batch in enumerate(self.val_loader):
                with torch.no_grad():
                    input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    with torch.autocast(device_type="cuda"):
                        feat1, feat2, _, _ = self.model(input_ids, attention_mask, mix=False)
                        losses = self.hard_loss(feat1, feat2, pairsimi)
                        val_loss += losses["instdisc_loss"]
            val_loss = val_loss.item()/(j+1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = step
                self.save_model(save_best=True)
    
def print_once(*args, **kwargs):
    #Print only once per GPU on DDP
    if dist.get_rank() == 0:  # Only print from rank 0
        print(*args, **kwargs)