import argparse
from Bio import SeqIO, AlignIO
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import SGD
from sklearn.metrics.pairwise import cosine_similarity
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile
import subprocess
from io import StringIO
from Bio.Align import MultipleSeqAlignment

import os
from datetime import datetime
import math

from transformer_infrastructure.attn_calc import get_attn_data

import pandas as pd
pd.set_option('display.max_rows', 2000)
import random

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt
#import seaborn as sns


from gymnasium import spaces

from time import time

# TODO: Fix the imports at some point
import torch.nn as nn
import torch.optim as optim


class ProteinMutationEnv(gym.Env):
    """
    A custom environment for mutating protein sequences using reinforcement learning.

    This environment is designed for the task of optimizing protein sequences towards a target sequence
    by applying possible mutations. It leverages a transformer model to evaluate the impact of mutations
    and uses reinforcement learning to guide the search for optimal sequences.

    Attributes:
        orig_ids (torch.Tensor): The original sequence IDs.
        target_ids (torch.Tensor): The target sequence IDs for optimization.
        possible_mutations (list): A list of possible mutations that can be applied to the sequence.
        model (transformers.PreTrainedModel): The transformer model used for sequence evaluation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the transformer model.
        device (torch.device): The device on which the model and tensors are allocated.
        int_limits (list, optional): A list of integer limits for the environment. Defaults to an empty list.
        int_ids (list, optional): A list of integer IDs for the environment. Defaults to an empty list.
        head_mask (torch.Tensor, optional): A mask for the transformer model heads. Defaults to None.
        baseline_aucs (list, optional): A list of baseline AUCs for evaluation. Defaults to an empty list.

    The environment defines an action space of possible mutations and an observation space representing the
    current state of mutations applied to the original sequence. It uses the transformer model to evaluate
    the impact of mutations and provides rewards based on the similarity of the mutated sequence to the target
    sequence and other criteria.
    """

    def __init__(self, orig_ids, target_ids, possible_mutations,  model, tokenizer, device, int_limits = [], int_ids = [], head_mask = None, baseline_aucs = []):

        super(ProteinMutationEnv, self).__init__()
        self.orig_ids = orig_ids
        self.target_ids = target_ids
        print("orig ids", orig_ids)
        print("target ids", target_ids)
        self.possible_mutations = possible_mutations
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.latest_reward = 0  
        self.mutstate =  [0] * len(self.possible_mutations)
        self.head_mask = head_mask
        self.baseline_aucs = baseline_aucs
        print("possible_mutations", self.possible_mutations)

        self.seqsim_history = []
        self.int_limits = int_limits
        self.int_ids = int_ids

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.possible_mutations))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.mutstate),), dtype=np.float32)
        self.current_ids = self.orig_ids.clone().detach()
        self.mutation_counter = 0  # Initialize the mutation counter
 
        # Set up initial sequence similarities, mse's, and target attentions and embeddings 
        # Happens once per episode, but all could be calculated beforehand. 
        # Anyway fast on GPU 
        orig_seq_tmp, orig_mask =  create_substitution_mask(self.mutstate, self.possible_mutations)
        target_seq_tmp, self.target_mask =  create_substitution_mask(self.mutstate, [(y, x) for x, y in self.possible_mutations])

        with torch.no_grad():

            # Adding orig to self, because could potentially also evaluate on similarity to the original sequence. 
            self.orig_mean_embedding, self.orig_attns = get_representation(self.model, self.orig_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True, head_mask = self.head_mask)
            self.target_mean_embedding, self.target_attns = get_representation(self.model, self.target_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True, head_mask = self.head_mask)

            self.starting_seqsim = F.cosine_similarity(self.orig_mean_embedding.to(device), target_mean_embedding.to(device)).item()
            self.best_seqsim = self.starting_seqsim


            self.orig_attns_aln =  filter_w_mask(self.orig_attns, torch.tensor(orig_mask, device = device))
            self.target_attns_aln =  filter_w_mask(self.target_attns, torch.tensor(self.target_mask, device = device))


            
            assert self.orig_attns_aln.shape == self.target_attns_aln.shape, f"Aligned attentions must be the same shape, {self.orig_attns_aln.shape} != {self.target_attns_aln.shape}"
 
            #self.starting_mse_self = np.mean((self.target_attns_aln.cpu().numpy() - self.orig_attns_aln.cpu().numpy()) ** 2).mean()
            self.starting_mse_self = torch.mean((self.target_attns_aln - self.orig_attns_aln) ** 2).item()


            #self.starting_mse_self = np.mean((self.target_attns.cpu().numpy() - self.orig_attns.cpu().numpy()) ** 2)
            self.lowest_mse_self = self.starting_mse_self

            if self.int_limits:
                spacer_len = 100

                spacer = torch.full((1, spacer_len), 5).to(device)
                orig_int_ids = torch.cat((self.orig_ids.unsqueeze(0),  spacer, self.int_ids), dim=1).to(device)
                target_int_ids = torch.cat((self.target_ids.unsqueeze(0),  spacer, self.int_ids), dim=1).to(device)


                orig_int_embeddings, orig_int_attns = get_representation(self.model, orig_int_ids, "t5", layers = [-1], output_attentions = True, head_mask = self.head_mask)   
                orig_int_attns_trim = trim_attns(attns = orig_int_attns, query_ids = self.orig_ids, query_int_ids = orig_int_ids, int_ids = self.int_ids, spacer_len = spacer_len, int_limits = self.int_limits)

                self.orig_int_attns_trim = filter_distance_heads(self.orig_int_attns_trim)

                target_int_embeddings, target_int_attns = get_representation(self.model, target_int_ids, "t5", layers = [-1], output_attentions = True, head_mask = self.head_mask)   
                self.target_int_attns_trim = trim_attns(attns = target_int_attns, query_ids = self.target_ids, query_int_ids = target_int_ids, int_ids = self.int_ids, spacer_len = spacer_len, int_limits = self.int_limits)
                self.target_int_attns_trim = filter_distance_heads(self.target_int_attns_trim)


                self.starting_mse_int = torch.mean((self.target_int_attns_trim - orig_int_attns_trim) ** 2).item()
                #self.starting_mse_int = np.mean((self.target_int_attns_trim.cpu().numpy() - orig_int_attns_trim.cpu().numpy()) ** 2)
                self.lowest_mse_int = self.starting_mse_int




    def step(self, action):
        # Apply mutation
        #print("STEP --------------------------------------------")
        # this is how we format it:
        # pass the state, then construct the sequence, then tokenize
        # mutstate: [0,1,0]. 
        # Action 1: ['AAAA', ''] # Deletion
        # Action 2: ['A', 'M']   # Substitution
        # Action 3: ['', 'GAMA'] # Insertion


        prev_seq = ''.join([element[flag] for element, flag in zip(self.possible_mutations, self.mutstate)])
        self.mutstate[action] = 1

        # Construct the sequence based on the mutstate flag 

        current_sequence, current_mask =  create_substitution_mask(self.mutstate, self.possible_mutations)
        self.current_ids = self.tokenizer(" ".join(current_sequence), return_tensors="pt", padding=True, truncation=True)["input_ids"][0].to(device)
        with torch.no_grad():

            current_mean_embedding, current_attns = get_representation(self.model, self.current_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True, head_mask = self.head_mask)
        seqsim = F.cosine_similarity(current_mean_embedding.to(device), self.target_mean_embedding.to(device)).item()
 


        current_attns_aln =  filter_w_mask(current_attns, torch.tensor(current_mask, device = device))

        assert current_attns_aln.shape == self.target_attns_aln.shape, "Aligned attentions must be the same shape"
        mse_self = torch.mean((self.target_attns_aln -  current_attns_aln) ** 2).item()


        if self.int_limits:
            spacer_len = 100

            spacer = torch.full((1, spacer_len), 5).to(device)

 
            current_int_ids = torch.cat((self.current_ids.unsqueeze(0), spacer, int_ids), dim=1).to(device)

            with torch.no_grad():
                current_int_embeddings, current_int_attns = get_representation(model, current_int_ids, "t5", layers = [-1], output_attentions = True, head_mask = self.head_mask)

            current_int_attns_trim = trim_attns(attns = current_int_attns, query_ids = self.current_ids, query_int_ids = current_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
            mse_int = torch.mean((self.target_int_attns_trim - current_int_attns_trim) ** 2).item()

            # Better     
            if mse_int < self.lowest_mse_int:
                reward_mse_int = self.lowest_mse_int - mse_int
                self.lowest_mse_int = mse_int
            # Worse
            else:
                reward_mse_int = 0

        # Better     
        if mse_self < self.lowest_mse_self:
           reward_mse_self = self.lowest_mse_self - mse_self
           self.lowest_mse_self = mse_self
        # Worse
        else:
           reward_mse_self = 0

        reward_ss = seqsim - self.best_seqsim

        # Only get a reward for improving on the cosine similarity
        
        # Worse
        if reward_ss <= 0:
            reward_ss = 0

        # Better
        else:
            self.best_seqsim = seqsim
      
        if self.int_limits:
            #print("REWARDS", reward_ss, reward_mse_int, reward_mse_self)
            reward = reward_mse_int * reward_mse_self  * reward_ss

        else:
            #print("REWARDS", reward_ss, reward_mse_self)
            reward = reward_mse_self  * reward_ss

        self.seqsim_history.append(seqsim)
        self.latest_reward = reward # not using this  
        self.mutation_counter += 1  # Increment mutation counter

        if self.baseline_aucs:
           #print(get_auc(self.seqsim_history))
           #print(get_auc(self.seqsim_history)[-1])
           #print(self.baseline_aucs)
           #print(self.baseline_aucs[len(self.seqsim_history)-1])
           #print("Compare to baseline")
           auc_reward = get_auc(self.seqsim_history)[-1] - self.baseline_aucs[len(self.seqsim_history) - 1]
           #print("AUC reward", auc_reward)
           if auc_reward > 0:
              auc_reward =  auc_reward
           else:
              #reward = 0
              auc_reward = 0

        else:
            auc_reward = []



        done = self.is_episode_done()

        if self.int_limits:
            return self.current_ids, reward, self.mutstate, self.best_seqsim, self.lowest_mse_self, self.lowest_mse_int, self.seqsim_history, auc_reward, done, {}

        else:
            # Add self.seqsim_history
            # self.baseline_auc
            
            return self.current_ids, reward, self.mutstate, self.best_seqsim, self.lowest_mse_self, self.seqsim_history, auc_reward, done, {}


    def pass_embeddings(self, input_ids, target_ids, model):
        # Unused sadly

        batch_input_ids = input_ids.unsqueeze(0)
        batch_target_ids = target_ids.unsqueeze(0)
 
        with torch.no_grad():
           embeddings = model.get_input_embeddings()(batch_input_ids)
    
        embeddings = embeddings.requires_grad_(True)
    
        outputs = model(inputs_embeds=embeddings, labels=batch_target_ids)
        logits= outputs.logits
        loss = outputs.loss
     

        # Getting the probabilities/gradients are not really necessary for now
        # They could be maybe used to prioritize actions to take 
        # Reshape for cross-entropy calculation: (batch_size * seq_len, vocab_size)
        reshaped_logits = logits.view(-1, logits.size(-1))
    
        # Flatten target_ids for the same reason: (batch_size * seq_len,)
        reshaped_target_ids = batch_target_ids.view(-1)
        # Calculate cross-entropy loss, without reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(reshaped_logits, reshaped_target_ids)
    
        # Reshape token_loss back to the sequence length for easier interpretation
        # This is where to mask out disorder before taking the mean to get the loss
        token_loss = token_loss.view(batch_target_ids.size())
        #probabilities = torch.exp(-token_loss)
    
        # Backward pass to get gradients
        loss.backward()
    
        # Clear grads
        model.zero_grad()
        for param in model.parameters():
            param.requires_grad = False
    
        outputs = None
        grads = embeddings.grad.detach().cpu()
        return embeddings, grads, loss

    def calculate_reward(self, current_ids, target_ids, model, tokenizer, device):
        # UNUSED 

        with torch.no_grad():

            target_mean_embedding, target_attns = get_representation(model, target_ids.unsqueeze(0), "t5", layers = [-1])
            orig_mean_embedding, orig_attns = get_representation(model, current_ids.unsqueeze(0), "t5", layers = [-1])


        cossim_orig_target = F.cosine_similarity(orig_mean_embedding.to(device), target_mean_embedding.to(device)).item()
        #print("mod_cossim", cossim_orig_target)

        # Define how to calculate the reward
        #tokens_current = tokenizer(" ".join(current_sequence), return_tensors="pt", padding=True, truncation=True)
        #input_ids = tokens_current["input_ids"].to(device)

        ##tokens_target = tokenizer(" ".join(target_sequence), return_tensors="pt", padding=True, truncation=True)
        #target_ids = tokens_target["input_ids"].to(device)

        #reward = cossim_orig_target.item()
        #print(reward)
        #embeddings, grads, loss = self.pass_embeddings(current_ids, target_ids, model)
        #reward = -loss.item() # Your reward calculation logic here
        return cossim_orig_target



    def is_episode_done(self):
        # Episode is done after 100 mutations
        # This is unused, just making all mutations
        # What if we make sure every round has a success, by eventually changing all amino acids
        return self.mutation_counter >= 1000

    def reset(self):
        print("RESET")
        self.mutstate = [0] * len(self.possible_mutations) 
        self.current_ids = self.orig_ids.clone().detach()
        self.mutation_counter = 0  # Reset the mutation counter
        self.latest_reward = 0 
        self.lowest_mse_self = self.starting_mse_self
        self.seqsim_history = []
        if self.int_limits:
            self.lowest_mse_int = self.starting_mse_int

        self.best_seqsim = self.starting_seqsim 
        return self.mutstate



    def render(self, mode='human'):
        # Print the current sequence

        if mode == 'human':
            outseq = self.tokenizer.decode(self.current_ids, skip_special_tokens=True)
            print(f"Reward: {self.latest_reward}, Current Sequence: {outseq}")
            #

    def close(self):
        pass


# Basic QNetwork
class QNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x



class PrioritizedReplayBuffer:

    def __init__(self, size):

        self.buffer = []
        self.size = size
        #self.success_threshold = success_threshold # Not necessary, already done


#
    def add(self, experience):
        # Experience is a tuple: (state, action, reward, next_state, done, success_metric)
        self.buffer.append(experience)
        # Keep buffer at its max size
        #if len(self.buffer) > self.size:
        #    self.buffer.pop(0)



    def sample(self, batch_size):
        # Calculate probabilities based on success metric
        epsilon = 0.01
        success_metrics = np.array([exp[2] for exp in self.buffer]) + epsilon
        probabilities = success_metrics / success_metrics.sum()



        # Sample based on probabilities
        chosen_indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        return [self.buffer[idx] for idx in chosen_indices]


class MABTracker:
    '''
    This is the multiarmed bandit
    '''

    def __init__(self, q_value_dict = {}):
        self.action_counts = 0
        self.q_value_dict = q_value_dict          
        self.step_log = []

    def update_q_values(self, action, auc_reward, episode, step):
        #step_size = 1.0 / step
        #self.q_value_dict[action] += step_size * (reward - self.q_value_dict[action])

        # So you get more bonus for taking a good action earlier
        if auc_reward > 0:
            print(action, possible_mutations[action][0], pos1[action], possible_mutations[action][1], pos2[action])
            print("AUC_reward", auc_reward)
            print("before update", self.q_value_dict[action])
      
            #self.q_value_dict[action] += (1 / (step + 1)) * (100 * auc_reward)# - self.q_value_dict[action])

            print("after update ", self.q_value_dict[action])
        self.step_log.append({
            'episode': episode,
            'step': step,
            'action': action,
            'aa1' : possible_mutations[action][0],
            'pos1' : pos1[action],
            'aa2' : possible_mutations[action][1],
            'pos2' : pos2[action],
            'reward': auc_reward,
            'cumulative_reward' : self.q_value_dict[action],
            'q_value' : self.q_value_dict[action]

        })




    def print_q_values(self):
        # Print cumulative rewards for each action
        for action, q_value in self.q_value_dict.items():
            print(f"Action {action} {possible_mutations[action]}: q_value = {q_value}")

    def print_q_table(self):
        intermediate2 = [(x, possible_mutations[x][0], y) for x, y in  list(self.q_value_dict.items())]
        df = pd.DataFrame(intermediate2, columns =['action', 'aa1', 'q_value'])
        
        return df



    def get_step_log(self):

        # Return a DataFrame with the log of all steps

        return pd.DataFrame(self.step_log) # , columns=['episode', 'step', 'action', 'reward'])




class ActionValueTracker:

    def __init__(self, cumulative_rewards = {}):
        self.cumulative_rewards = {}
        self.action_counts = {}
        self.step_log = []

    

    def update(self, action, reward, episode, step):
 
        if action not in self.cumulative_rewards:
            self.cumulative_rewards[action] = 0
            self.action_counts[action] = 0
        self.cumulative_rewards[action] += reward
        self.action_counts[action] += 1

        # TODO: add pos1, pos2, possible_mutations to self
        # Log this step and the current cumulative rewards
        self.step_log.append({
            'episode': episode,
            'step': step,
            'action': action,
            'aa1' : possible_mutations[action][0],
            'pos1' : pos1[action],
            'aa2' : possible_mutations[action][1],
            'pos2' : pos2[action],
            'reward': reward,
            'cumulative_reward' : self.cumulative_rewards[action]
        })



    def get_average_reward(self, action):
        if action not in self.cumulative_rewards:
            return 0
        return self.cumulative_rewards[action] / self.action_counts[action]


    def get_action_values(self):
        # Return a dictionary of action values
        action_values = {}
        for action in self.cumulative_rewards:
            action_values[action] = self.get_average_reward(action)
        return action_values

    def print_cumulative_rewards(self):
        # Print cumulative rewards for each action
        for action, cumulative_reward in self.cumulative_rewards.items():
            print(f"Action {action} {possible_mutations[action]}: Cumulative Reward = {cumulative_reward}")

    def cumulative_rewards_table(self):
        intermediate2 = [(x, possible_mutations[x][0], y) for x, y in  list(self.cumulative_rewards.items())]
        df = pd.DataFrame(intermediate2, columns =['action', 'aa1', 'cumulative_reward'])
        return df

    def get_step_log(self):

        # Return a DataFrame with the log of all steps

        return pd.DataFrame(self.step_log) # , columns=['episode', 'step', 'action', 'reward'])



def generate_random_protein(n):
    '''
    Given a length n, generate a protein of that length
    '''
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    protein_sequence = ''.join(random.choice(amino_acids) for _ in range(n))
    return protein_sequence
   

def apply_actions_in_order(actions, possible_mutations, target_ids, model, tokenizer, device, int_ids = torch.tensor([]), spacer_len = 0, int_limits = [], head_mask = None):
 
    seqsims = []    
    seqs = []
    self_attn_mses = []
    int_attn_mses = []
    mutstate = [0] * len(possible_mutations)
    spacer = torch.full((1, spacer_len), 5).to(device)
    # Get the mask for simple substitutions in the target sequenc

    target_seq_tmp, target_mask =  create_substitution_mask(mutstate, [(y, x) for x, y in possible_mutations])

    with torch.no_grad():

        target_mean_embedding, target_attns  = get_representation(model, target_ids, "t5", layers = [-1], output_attentions = True, head_mask = head_mask)#.to(device)
        target_attns_aln =  filter_w_mask(target_attns, torch.tensor(target_mask, device = device)) # Do this at the start, being repetitive

        if int_limits:
                target_int_ids = torch.cat((target_ids, spacer, int_ids), dim=1).to(device) 
                target_int_embeddings, target_int_attns = get_representation(model, target_int_ids, "t5", layers = [-1], output_attentions = True, head_mask = head_mask)


    for action in actions:

        mutstate[action] = 1
 
        # Construct the sequence based on the mutstate flag 
        current_sequence_prevway = ''.join([element[flag] for element, flag in zip(possible_mutations, mutstate)])
        current_sequence, current_mask =  create_substitution_mask(mutstate, possible_mutations) 
        assert current_sequence_prevway == current_sequence, "If these aren't the same, check create_substitution_mask function"
   
        seqs.append(current_sequence)

        current_ids = tokenizer(" ".join(current_sequence), return_tensors="pt", padding=True, truncation=True)["input_ids"][0].to(device)


        with torch.no_grad():
            current_mean_embedding, current_attns  = get_representation(model, current_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True, head_mask = head_mask)#.to(device)
            current_attns_aln =  filter_w_mask(current_attns, torch.tensor(current_mask, device = device))
            assert current_attns_aln.shape == target_attns_aln.shape, "Aligned attentions must be the same shape"

            mse_self = torch.mean((target_attns_aln - current_attns_aln) ** 2).item()

            self_attn_mses.append(mse_self)
            if int_limits:
                current_int_ids = torch.cat((current_ids.unsqueeze(0), spacer, int_ids), dim=1).to(device)
                get_rep_time = time()
                current_int_embeddings, current_int_attns = get_representation(model, current_int_ids, "t5", layers = [-1], output_attentions = True, head_mask = head_mask)
                current_int_attns_trim = trim_attns(attns = current_int_attns, query_ids = current_ids, query_int_ids = current_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
                mse_int = torch.mean((target_int_attns - current_int_attns) ** 2).item()

                int_attn_mses.append(mse_int)

        seqsim = F.cosine_similarity(current_mean_embedding.to(device), target_mean_embedding.to(device)).item()
        seqsims.append(seqsim)
    if int_limits:
        return seqsims, seqs, self_attn_mses, int_attn_mses
    else:
        return seqsims, seqs, self_attn_mses
 

def get_auc(y_values):
    """
    Calculate the cumulative AUC up until each step for an ordered list of y-values,
    assuming uniform spacing between each step on the x-axis.
    
    Parameters:
    - y_values: List of y-values (ordered).
    
    Returns:
    - List of cumulative AUC values up until each step.
    """
    # Initialize the list to store cumulative AUC values, starting with 0
    cumulative_auc = [0]
    
    # Iterate over the y_values to calculate the cumulative AUC up to each step
    for i in range(1, len(y_values)):
        # Calculate the area of the trapezoid formed by two consecutive points
        step_auc = ((y_values[i-1] + y_values[i]) / 2)
        # Add the step AUC to the cumulative total
        cumulative_auc.append(cumulative_auc[-1] + step_auc)
    
    return cumulative_auc

def process_episode_actions(total_rewards, episodes, possible_mutations, target_ids, model, tokenizer, device, int_fasta_file=None, int_ids=None, int_limits=None, head_mask=None):
    for episode in episodes:
        # Get actions for this episode sorted by rank
        episode_actions = total_rewards[total_rewards['episode'] == episode]
        actions_list = episode_actions['action'].tolist()
        # Apply actions in order and update DataFrame
        if int_fasta_file is not None:
            similarity_score, out_seqs, self_attn_mses, int_attn_mses = apply_actions_in_order(
                actions_list, possible_mutations, target_ids, model, tokenizer, device, 
                int_ids=int_ids, spacer_len=100, int_limits=int_limits, head_mask=head_mask
            )
        
            total_rewards.loc[total_rewards['episode'] == episode, 'int_attn_mses'] = int_attn_mses
        else:
            similarity_score, out_seqs, self_attn_mses = apply_actions_in_order(
                actions_list, possible_mutations, target_ids, model, tokenizer, device, 
                head_mask=head_mask
            )
        total_rewards.loc[total_rewards['episode'] == episode, 'self_attn_mses'] = self_attn_mses
        total_rewards.loc[total_rewards['episode'] == episode, 'similarity_score'] = similarity_score
        total_rewards.loc[total_rewards['episode'] == episode, 'out_seq'] = out_seqs
   
    return total_rewards



def plot_attention_heatmaps(attention_weights, basename, identifier):

    """

    Plots a series of heatmaps for attention weights of a T5 model.

    

    :param attention_weights: A tensor or array of shape 

                              (num_layers, num_heads, sequence_length, sequence_length)

                              containing the attention weights.

    """

    
    num_layers, num_heads, _, _ = attention_weights.shape
    

    fasta_id = basename.split("/")[-1]
    outdir_images = f'{basename}_outdir/{fasta_id}_images/'

    os.makedirs(outdir_images, exist_ok=True) 

    rows_per_layer = math.ceil(num_heads / 4)



    for layer in range(num_layers):

        fig, axs = plt.subplots(rows_per_layer, 4, figsize=(20, rows_per_layer * 4), squeeze = False)  # Adjust figsize as needed

        fig.suptitle(f'Layer {layer + 1} Attention patterns for {identifier}')

        for head in range(num_heads):

            # Extract the attention weights for the current layer and head

            attention_matrix = attention_weights[layer, head]

            # Plotting
            ax = axs[head // 4, head % 4]  # Calculate row and column index

            im = ax.imshow(attention_matrix, aspect='auto', cmap='viridis')

            #im = axs[head].imshow(attention_matrix, aspect='auto', cmap='viridis')
            ax.set_title(f'Head {head + 1}')

            ax.set_xlabel('Seq Pos')

            ax.label_outer()


        plt.show()

        plt.savefig(f'{outdir_images}/{fasta_id}_Layer_{layer+1:02d}_{identifier}.pdf')

        plt.close() 



def is_strict_distance_head(attention_matrix, distance_thresholds=[1, -1, 2, -2]):

    seq_length = attention_matrix.shape[-1]
    attention_matrix = attention_matrix > 0.1  # Assuming attention > 0.1 indicates focus

    for dist in distance_thresholds:
        if dist > 0:
            shifted_matrix = torch.cat([torch.zeros((seq_length, dist)), torch.eye(seq_length)[:, :-dist]], dim=1)
        elif dist < 0:

            shifted_matrix = torch.cat([torch.eye(seq_length)[:, -dist:], torch.zeros((seq_length, -dist))], dim=1)
        else:
            continue  # Skip distance 0

        if torch.all((attention_matrix == shifted_matrix.to(device))): # | (attention_matrix == 0)):

            return True
    return False

 


def get_distance_head_mask(attention_tensor):

    # Identify heads to keep
    # Initialize a mask to keep track of heads to keep for each layer
    heads_to_keep = torch.zeros((attention_tensor.size(0), attention_tensor.size(1)), dtype=torch.bool)  # For 24 layers and 30 heads


    # Identify heads that don't strictly follow a distance pattern
    for layer_idx in range(attention_tensor.shape[0]):
        for head_idx in range(attention_tensor.shape[1]):
            if not is_strict_distance_head(attention_tensor[layer_idx, head_idx]):
                heads_to_keep[layer_idx, head_idx] = True
            else:
                print("this is a distance head?", layer_idx, head_idx, attention_tensor[layer_idx, head_idx])


    # Example: Print the number of heads to keep per layer
    print("Heads to keep per layer:", heads_to_keep.sum(dim=1).tolist())



    # OK, big question here, are distance heads always the same for different sequence? Probably yes
 

    #filtered_tensor = torch.where(expanded_mask, tensor, torch.tensor(0.))

    return(heads_to_keep)



def attn_stats(tensor, basename):
    
    # Calculate the variance and mean along the last two dimensions
   
 
    variance = torch.var(tensor, dim=(-2, -1), unbiased=False)  # Shape [24, 30]
    
    mean = torch.mean(tensor, dim=(-2, -1))  # Shape [24, 30]
    
    
    # Set up the matplotlib figure
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    
    
    # Histogram of variances
    
    axs[0].hist(variance.flatten().cpu().numpy(), bins=30, color='blue', alpha=0.7)
    
    axs[0].set_title('Distribution of Variances')
    
    axs[0].set_xlabel('Variance')
    
    axs[0].set_ylabel('Frequency')
    print(mean.shape)
    print(variance.shape) 
    
    print(mean.flatten().cpu().numpy().tolist())
    print(variance.flatten().cpu().numpy().tolist())
 
    # Histogram of means
    
    axs[1].hist(mean.flatten().cpu().numpy(), bins=30, color='green', alpha=0.7)
    
    axs[1].set_title('Distribution of Means')
    
    axs[1].set_xlabel('Mean')
    
    axs[1].set_ylabel('Frequency')
    
    
    # Scatter plot of mean vs. variance for a sample of n x n matrices
    
    
    
    axs[2].scatter(mean.flatten().cpu().numpy(), variance.flatten().cpu().numpy(), alpha=0.5)
    
    axs[2].set_title('Mean vs. Variance')
    
    axs[2].set_xlabel('Mean')
    
    axs[2].set_ylabel('Variance')
    
    
    
    plt.tight_layout()
    
    plt.savefig(f'{basename}_attn_stats_plot.png')
    plt.close()
    # Flatten the tensors and convert them to numpy
    
    variance_np = variance.flatten().cpu().numpy()
    
    mean_np = mean.flatten().cpu().numpy()
    
    
    
    # Create DataFrame and save as CSV
    
    df = pd.DataFrame({
    
        'dim1': torch.arange(tensor.shape[0]).repeat_interleave(tensor.shape[1]).numpy(),
    
        'dim2': torch.arange(tensor.shape[1]).repeat(tensor.shape[0]).numpy(),
    
        'mean': mean_np,
    
        'variance': variance_np
    
    })
    
    df.to_csv(f'{basename}_attn_stats.csv')


    # Find the indices where the variance is zero
    zero_variance_indices = (variance == 0)

    #print(zero_variance_indices)
    print("zero variance indices = ", zero_variance_indices.sum().item())

    # Replace the corresponding n x n matrices with zeros

    tensor[zero_variance_indices] = 0
    return(tensor)
 


# The mask is a list, change the variables
def filter_w_mask(tensor_attns, tensor_mask):
 
        len_tensor = tensor_attns.shape[3]
        tensor_mask_a = tensor_mask.view(1, 1, len_tensor, 1).expand(-1, -1, -1, len_tensor)
        tensor_mask_b = tensor_mask.view(1, 1, 1, len_tensor).expand(-1, -1, len_tensor, -1)

        combined_tensor_mask = (tensor_mask_a * tensor_mask_b) == 1
        combined_tensor_mask = combined_tensor_mask.expand_as(tensor_attns)
        filtered_tensor = tensor_attns[combined_tensor_mask]
        return(filtered_tensor)





def create_substitution_mask(mutstate, actions):

    # Initialize output sequence and mask

    output_sequence = ''

    mask = []



    # Apply each action from mutstate

    for state, action in zip(mutstate, actions):

        before, after = action



        # Check if the action is a substitution based on the length of 'before' and 'after'
        is_substitution = len(before) == len(after)  == 1



        # If the action is taken

        if state == 1:

            output_sequence += after

            mask += [1 if is_substitution else 0] * len(after)

        else:

            # If the action is not taken, still append 'before' to the output sequence

            output_sequence += before

            

            # Append to mask based on whether it's a substitution

            mask += [1 if is_substitution else 0] * len(before)



    return output_sequence, mask





# get rank based on current Q-value

actions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]







# Probability of moving an action earlier



move_earlier_prob = 0.3  # 30% chance







# Process each action starting from the second one


def get_neworder(actions, move_earlier_prob = 0.1):
    '''
    Make sure it's being reset each time
    '''
    print("STARTING ACTIONS", actions)
    for i in range(1, len(actions)):

        # Move 1% of the mutations to the start each time. 
        if random.random() < move_earlier_prob:

            # Randomly choose a new position that is earlier than the current one
            # Actually move just much earlier in the sequence. 
            # Like in the first 10
            #new_position = random.randint(0, i-1)
            new_position = 0 # random.randint(0, 10)
            # Move the action to its new position
            actions.insert(new_position, actions.pop(i))

    return(actions)

def get_neworder2(actions, number_to_advance = 10):
    '''
    Make sure it's being reset each time
    '''
    
    promoted = random.sample(actions, number_to_advance)
    remaining_actions = [num for num in actions if num not in promoted]
    new_actions = promoted + remaining_actions
 

    #print("STARTING ACTIONS", actions)
    #for i in range(1, len(actions)):

    #    # Move 1% of the mutations to the start each time. 
    #    if random.random() < move_earlier_prob:

            # Randomly choose a new position that is earlier than the current one
            # Actually move just much earlier in the sequence. 
            # Like in the first 10
            #new_position = random.randint(0, i-1)
    #        new_position = 0 # random.randint(0, 10)
            # Move the action to its new position
    #        actions.insert(new_position, actions.pop(i))

    return(new_actions)




def parse_arguments():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Optimize protein sequence embeddings with a model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the pretrained model.")

    parser.add_argument("-f", "--fasta_file", type=str, required=True, help="Path to the FASTA file with two sequences: the query and the target.")

    parser.add_argument("-i", "--int_fasta_file", type=str, required=False, help="A fasta file with a single target binding sequence")



    parser.add_argument("-a", "--aln_file", type=str, help="Path to a FASTA alignment file with two sequences: the query and the target.")

    #parser.add_argument("-s", "--steps", type=int, help="Number of mutations to make.")

    parser.add_argument("-c", "--cnn_path", type=str, required=False, help="Path to disorder predictor model, SETH_CNN.pt")

    parser.add_argument("-e", "--episodes", type=int, required=False, help="Number of episodes", default = 50)
  
    parser.add_argument("-emab", "--episodes_mab", type=int, required=False, help="Number of episodes of refining multi-armed bandit", default = 0)
  
 
    parser.add_argument("-r", "--reward_strat", type=str, nargs='+', choices=['ss', 'as', 'is'], default=['ss', 'choice2'], help="One or more choices,.")

 
    parser.add_argument("-rs", "--random_start", action = "store_true", help = "Flag if want to start from random protein sequence")

    # The arguments are toward figuring out how to "align" the attention networks between two sequences
    # Attention similarity is a strong signal, but position changes mask structure changes
    parser.add_argument("-x", "--xgap", action = "store_true", help = "Flag if want to replace gaps with X's")

    parser.add_argument("-d", "--dgap", action = "store_true", help = "Flag if want to duplicate insertions and gaps between sequences")




    return parser.parse_args()

def get_attn_data(model, tokenizer, tokens, min_attn = 0.1,  max_seq_len=1024, model_type = "bert"):


    with torch.no_grad():
        attns = model(inputs)[-1]
    if model_type == "bert":

      front_trim = 1
      end_trim = 1

    elif model_type == "t5" or model_type == "gpt2":

      front_trim = 0
      end_trim = 1

    attns = [attn[:, :, front_trim:-end_trim, front_trim:-end_trim] for attn in attns]
    attns = torch.stack([attn.squeeze(0) for attn in attns])
    attns = attns.tolist()
    return(attns)




def do_align_new(seqrecordlist, filename):

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as temp_fasta:
        SeqIO.write(seqrecordlist, temp_fasta, "fasta")
        temp_fasta_name = temp_fasta.name
    clustalo_exe = "clustalo"  # Adjust if necessary
    cline = ClustalOmegaCommandline(clustalo_exe, infile=temp_fasta_name, outfile=f"{filename}.output.aln", verbose=True, auto=True, force = True) # Comput up with outfile based on fasta
    stdout, stderr = cline()
    alignment = AlignIO.read(f"{filename}.output.aln", "fasta")
    return(alignment)

def map_positions(seq1, seq2):
    """
    Create a dictionary mapping positions in sequence 1 to positions in sequence 2. Sequence should be aligned using gaps ("-")
    """
    position_map = {}
    pos_seq1 = 0  # Counter for non-gap positions in seq1
    pos_seq2 = 0  # Counter for non-gap positions in seq2

    for char_seq1, char_seq2 in zip(seq1, seq2):

        if char_seq1 != '-':
            if char_seq2 != '-':
                position_map[pos_seq1] = pos_seq2
            pos_seq1 += 1
        if char_seq2 != '-':
            pos_seq2 += 1

    return position_map


def retrieve_aa_embeddings(model_output, model_type, layers = [-4,-3,-2,-1]):

    '''

    Get the amino acid embeddings for each sequences

    Pool layers by concatenating selection of layers

    Return shape: (numseqs, length of longest sequence, 1024*numlayers)

    Takes: 

       model_output: From sequence encoding

       layers (list of ints): By default, pool final four layers of model



    Return shape (numseqs x longest seqlength x (1024 * numlayers)

    Note: If the output shape of this function is [len(seqs), 3, x], make sure there are spaces between each amino acid

    The "3" corresponds to CLS,seq,END 

    '''

    # Get all hidden states



    hidden_states = model_output.hidden_states

    # Concatenate hidden states into long vector



    # Either layers or heads

    if layers is not None:
        aa_embeddings = torch.cat(tuple([hidden_states[i] for i in layers]), dim=-1)

    if model_type == "bert":
      front_trim = 1
      end_trim = 1

    elif model_type == "t5" or model_type == "gpt2":
      front_trim = 0
      end_trim = 1

    else:

       print("Model type required to extract aas. Currently supported bert and t5")

       return(0)



    aa_embeddings = aa_embeddings[:,front_trim:-end_trim,:]

    return(aa_embeddings, aa_embeddings.shape)





def get_representation(model, input_ids, model_type, layers, mask = None, output_attentions = False, remove_0var_attns = False, head_mask = None):

    with torch.no_grad():
        model_output =  model.encoder(input_ids=input_ids, output_attentions = output_attentions)

        if output_attentions == True:
             attns = model_output[-1]

    aa_embeddings, aa_shape = retrieve_aa_embeddings(model_output, model_type = model_type, layers = layers)

    aa_embeddings = aa_embeddings.to('cpu')

    if mask is not None:

       bool_mask = torch.tensor(mask, dtype=torch.bool)

       masked_embeddings = aa_embeddings[:, ~bool_mask]


       sequence_representation = torch.mean(masked_embeddings, dim = 1)

    else:

        sequence_representation = torch.mean(aa_embeddings, dim = 1)#np.mean(aa_embeddings, axis = 1)

    if output_attentions == True:
        # Remove attention from <CLS> (first in BERT) and <SEP> (last) token
        if model_type == "bert":
          front_trim = 1
          end_trim = 1
    
        elif model_type == "t5" or model_type == "gpt2":
          front_trim = 0
          end_trim = 1
    
        else:
    
           print("Model type required to extract attentions. Currently supported bert and t5")
    
           return(0)
        #print("pretrim", len(attns)) # for t5 this a list of 24 layers, each with 32 heads
        attns = [attn[:, :, front_trim:-end_trim, front_trim:-end_trim] for attn in attns]
        #print("posttrim", attns[0].shape, len(attns))
        attns = torch.stack([attn.squeeze(0) for attn in attns])
        #print("poststack", attns.shape)
        #attns = attns.tolist()
        if head_mask is not None:
           #print(head_mask)
           expanded_mask = head_mask.unsqueeze(-1).unsqueeze(-1).expand_as(attns)
           attns = torch.where(expanded_mask, attns, torch.tensor(0.).to(device))



        return(sequence_representation, attns)
 
    else:
        return(sequence_representation)
 
def update_target_network(primary_network, target_network):

    target_network.load_state_dict(primary_network.state_dict())





def generate_differences_and_positions(seq1, seq2):

    if len(seq1) != len(seq2):

        raise ValueError("Sequences must be of the same length for alignment")



    # Generate initial tuples and positions
    differences, positions1, positions2 = [], [], []
    pos1, pos2 = 0, 0  # Position counters



    for char1, char2 in zip(seq1, seq2):
        if char1 == '-' and char2 == '-':
            continue



        elif char1 == '-' or char2 == '-':

            if char1 == '-':

                differences.append(('', char2))
                positions1.append('NA')
                positions2.append(pos2)
                pos2 += 1



            else:

                differences.append((char1, ''))
                positions1.append(pos1)
                positions2.append('NA')
                pos1 += 1

        else:

            differences.append((char1, char2))
            positions1.append(pos1)
            positions2.append(pos2)
            pos1 += 1
            pos2 += 1


    # Combine consecutive tuples and positions
    combined_diffs, combined_pos1, combined_pos2 = [], [], []
    i = 0



    while i < len(differences):

        current_diff = differences[i]
        current_pos1 = positions1[i]
        current_pos2 = positions2[i]

        while i + 1 < len(differences) and (current_diff[0] == '' and differences[i + 1][0] == '' or
                                           current_diff[1] == '' and differences[i + 1][1] == ''):
            i += 1

            current_diff = (current_diff[0] + differences[i][0], current_diff[1] + differences[i][1])


        combined_diffs.append(current_diff)
        combined_pos1.append(current_pos1)
        combined_pos2.append(current_pos2)
        i += 1


    return combined_diffs, combined_pos1, combined_pos2


def get_total_attn(attns):

    attns_sum = attns.sum().item()

    return(attns_sum)

def trim_attns(attns, query_ids, query_int_ids, int_ids,  spacer_len = 0, int_limits = []):

    offset = query_int_ids.shape[1] - int_ids.shape[1]
    #print("offset", offset, spacer.shape[1], query_ids.shape[1])
    if not int_limits:
        #print(query_int_ids.shape[1], offset, int_ids.shape[1])
        int_limits  = [query_int_ids.shape[1] - int_ids.shape[1], query_int_ids.shape[1]]

    int_limits_offset = [x + offset for x in int_limits]

    # What about the end characters. 
    #print("query_ids.shape[0]", query_ids.shape[0])
    #print("attns.shape", attns.shape)

    assert torch.equal(int_ids[:, int_limits[0]: int_limits[1] + 1], query_int_ids[:, int_limits_offset[0]: int_limits_offset[1] + 1]), "something wrong with int_limits"

    attns_trim = attns[:, :, :query_ids.shape[0] + 1, int_limits_offset[0] : int_limits_offset[1] + 1]
    #print("attns_trim.shape", attns_trim.shape)
    #total_attns = attns.sum(dim=(0, 1))
    #attns_trim = total_attns[int_limits_offset[0]: int_limits_offset[1] + 1, int_limits_offset[0] : int_limits_offset[1] + 1]
    #print(query_int_attns_trim.shape)
    #attns_sum  = attns_trim.sum().item()
    return(attns_trim) 



if __name__ == "__main__":

    args = parse_arguments()
    fasta_file = args.fasta_file
    model_path = args.model_path
    int_fasta_file = args.int_fasta_file

    cnn_path = args.cnn_path
    aln_file = args.aln_file
    reward_strat = args.reward_strat
    random_start = args.random_start

    num_episodes = args.episodes
    num_episodes_mab = args.episodes_mab

    xgap = args.xgap
    dgap = args.dgap
    # 'ss', 'sa', 'ia'


    #xgap = False
    if not fasta_file:
       if not aln_file:
          print("Provide fasta or alignment")
          exit(1)

     

    model = T5ForConditionalGeneration.from_pretrained(model_path, output_hidden_states = True).eval()
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model_config = AutoConfig.from_pretrained(model_path)
    model_type = model_config.model_type
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        print(f"{token}: {token_id}")

 


    #cpu_only = False
    #if cpu_only == True:
    #    device = "cpu"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    
 
    model = model.to(device)

    layers = [-1]
    # Ensure model parameters are not updated

    for param in model.parameters():
        param.requires_grad = False

    # Read sequences from FASTA

    if fasta_file:

        with open(fasta_file, "r") as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
 
        path, filename = os.path.split(fasta_file)

        # Format the current date and time as YYYY-MM-DD

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3] # Only need 3 digits of milliseconds instead of 6

        output_dir = os.path.join(path, f"{filename}_{date_str}_ep{num_episodes}_epmab{num_episodes_mab}")

        os.makedirs(output_dir, exist_ok=True)

        outfile_base = os.path.join(output_dir, filename)

        params_file = os.path.join(output_dir, "input_params.txt")

        # Write parameters to the file
        
        with open(params_file, "w") as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

        if random_start: 
            target_seqrecord = sequences[0]
            target_seq_nospaces = (str(target_seqrecord.seq))
            target_name = target_seqrecord.id

            orig_seqrecord = SeqRecord(Seq(generate_random_protein(len(target_seqrecord.seq))), id="random_seq")
            orig_seq_nospaces = (str(orig_seqrecord.seq))
            orig_name = orig_seqrecord.id


            alignment = AlignIO.MultipleSeqAlignment([target_seqrecord, orig_seqrecord])
            AlignIO.write(alignment, f"{outfile_base}.randstart.aln", "fasta")




        else:
            orig_seqrecord = sequences[0]
            target_seqrecord = sequences[1]
    
            orig_name = orig_seqrecord.id
            target_name = target_seqrecord.id
    
            orig_seq_nospaces = (str(orig_seqrecord.seq))
            target_seq_nospaces = (str(target_seqrecord.seq))
    
            if not aln_file:
                starting_alignment = do_align_new([target_seqrecord, orig_seqrecord], outfile_base)
                alignment = starting_alignment
    
            else:
                print(f"Using existing alignment {aln_file}")
                alignment = AlignIO.read(aln_file, "fasta")
                orig_seqrecord = alignment[0]
                target_seqrecord = alignment[1]
                orig_name = orig_seqrecord.id
                target_name = target_seqrecord.id
 
                # Really should fix so that the alignment is in the same order as the fasta
                alignment = AlignIO.MultipleSeqAlignment([target_seqrecord,orig_seqrecord])
    
    
            print("query:  ", alignment[0].seq)
            print("target: ", alignment[1].seq)
    if xgap == True:
        target_seq_nospaces = alignment[0].seq.replace("-", "X")
        orig_seq_nospaces = alignment[1].seq.replace("-", "X")
    


    AlignIO.write(alignment, f"{outfile_base}.aln", "fasta")


    print("orig_seqrecord", orig_seqrecord)
    print("target_seqrecord", target_seqrecord)

    print(orig_seq_nospaces)
    print(target_seq_nospaces)  

    tokens_orig = tokenizer(" ".join(orig_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
    orig_ids = tokens_orig["input_ids"].to(device)
    tokens_target = tokenizer(" ".join(target_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
    target_ids = tokens_target["input_ids"].to(device)

    print("orig_ids", orig_ids)
    print("target_ids", target_ids)




    with torch.no_grad():
    
        target_mean_embedding, target_attns = get_representation(model, target_ids, model_type, layers, output_attentions = True)
        orig_mean_embedding, orig_attns = get_representation(model, orig_ids, model_type, layers, output_attentions = True) 
 
    cossim_orig_target = F.cosine_similarity(orig_mean_embedding.to(device), target_mean_embedding.to(device))
    
    head_mask = get_distance_head_mask(target_attns)
    print("head_mask", head_mask)
    head_mask = head_mask.to(device)

    with torch.no_grad():
    
        target_mean_embedding, target_attns = get_representation(model, target_ids, model_type, layers, output_attentions = True, head_mask = head_mask)
        orig_mean_embedding, orig_attns = get_representation(model, orig_ids, model_type, layers, output_attentions = True, head_mask = head_mask) 

    outdir = f'{outfile_base}_outdir/'

    os.makedirs(outdir, exist_ok=True) 

    # Works, but slow for every run.  
    #plot_attention_heatmaps(target_attns.cpu(), outfile_base, "target_attns")
    #plot_attention_heatmaps(orig_attns.cpu(), outfile_base, "orig_attns")

  

    #target_attns = filter_distance_heads(target_attns)
    #orig_attns = filter_distance_heads(orig_attns)

   
    attn_stats(target_attns, outfile_base) 
    attn_stats(orig_attns, outfile_base)


    int_limits = []
    int_ids = []

    if int_fasta_file:
        with open(int_fasta_file, "r") as handle:
            int_seqrecord = list(SeqIO.parse(handle, "fasta"))[0]
            int_seq_nospaces = str(int_seqrecord.seq)
            int_name = int_seqrecord.id
            int_tokens = tokenizer(" ".join(int_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
            int_ids = int_tokens["input_ids"].to(device)

            spacer_len = 100
            spacer = torch.full((1, spacer_len), 5).to(device)
            # Is there something off with the shapes here? why does trim_attns not work here?
            target_int_ids = torch.cat((target_ids, spacer, int_ids), dim=1).to(device)
            print("shapes", target_ids.shape, int_ids.shape, target_int_ids.shape)
            # target_int_attns are the interface attentions that the binder has

            orig_int_ids = torch.cat((orig_ids, spacer, int_ids), dim=1).to(device)
            
     
            with torch.no_grad():
                target_int_embeddings, target_int_attns = get_representation(model, target_int_ids, model_type, layers, output_attentions = True, head_mask = head_mask)
                orig_int_embeddings, orig_int_attns = get_representation(model, orig_int_ids, model_type, layers, output_attentions = True, head_mask = head_mask)
     

            #target_attns_arr = np.sum(target_attns.cpu().numpy(), axis=(2, 3))
            #orig_attns_arr = np.sum(orig_attns.cpu().numpy(), axis=(2, 3))
            #mse_self = np.mean((target_attns_arr - orig_attns_arr) ** 2)
 
            # Add an argument for this
            int_limits = [9, 109]
            if len(int_limits) == 0:
                int_limits = [0, len(int_ids)] # The entire length of the interactor

        
            target_int_attns_trim = trim_attns(attns = target_int_attns, query_ids = target_ids, query_int_ids = target_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
            print("target_int_attns.shape", target_int_attns_trim.shape)
         
            orig_int_attns_trim = trim_attns(attns = orig_int_attns, query_ids = orig_ids, query_int_ids = orig_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
            print(orig_int_attns_trim.shape)
        
         
            # Plotting
            
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            # Heatmap for the first aggregated array
            
            #axes[0].imshow(diff, cmap='coolwarm', vmin = -20, vmax= 20,  interpolation='nearest')
            #axes[0].set_title('Aggregated Array 1')
            #axes[0].set_xlabel('Dimension 2 (size 16)')
            #axes[0].set_ylabel('Dimension 1 (size 32)')
            
            
            
            # Heatmap for the second aggregated array
            #axes[1].imshow(diff, cmap='viridis', interpolation='nearest')
            #axes[1].set_title('Aggregated Array 2')
            #axes[1].set_xlabel('Dimension 2 (size 16)')
            #axes[1].set_ylabel('Dimension 1 (size 32)')
        
            # Display colorbar
            #cbar = fig.colorbar(axes[0].imshow(diff, cmap='viridis', interpolation='nearest'), ax=axes.ravel().tolist())
            #cbar.set_label('Sum of values in 80x80 block')
            
            # Save the figure
            #plt.tight_layout()
            #plt.show() 
            #plt.savefig(f'{fasta_file}_aggregated_arrays.png', dpi=300)
        
     
    

    alignment_map = map_positions(alignment[-1].seq, alignment[0].seq)
    print(f"cossim_orig_target: {cossim_orig_target}")
    print("alignment", alignment)
    print(alignment_map)
    # what we an do here with disorder is both convert the sequences to masked, as well as don't take any actions for disorder masked sequence
    mutannot = []
    # What I need is tuples mapping numeric actions to ex. insertions + deletions

    # So the examined protein is a concatenation of all the actions, translated to insertions, deletions, or mutations

    possible_mutations, pos1, pos2 = generate_differences_and_positions(alignment[-1], alignment[0])

    print(str(alignment[0].seq))
    print(str(alignment[1].seq))
    print("possible_mutations", possible_mutations)


    if xgap == True:

        # BEFORE
        # AB-CDE
        # TUVW--
        
        # AFTER
        # ABXCDE
        # TUVWXX

        # Still give big boosts from X to to real sequence
        possible_mutations_tmp = []
        for x in possible_mutations:
           if x[0] == "":
          
              possible_mutations_tmp.append(("X"*len(x[1]), x[1]))
           elif x[1] == "":
              possible_mutations_tmp.append((x[0], "X"*len(x[0])))
           else:
              possible_mutations_tmp.append(x)

        possible_mutations = possible_mutations_tmp 

    if dgap == True:
        # BEFORE
        # AB-CDE
        # TUVW--
        
        # AFTER
        # ABVCDE
        # TUVWDE

        possible_mutations_tmp = []
        for x in possible_mutations:
           if x[0] == "":
          
              possible_mutations_tmp.append((x[1], x[1]))
           elif x[1] == "":
              possible_mutations_tmp.append((x[0], x[0]))
           else:
              possible_mutations_tmp.append(x)

        possible_mutations = possible_mutations_tmp 

       
      
        orig_seq_nospaces = "".join([x[0] for x in possible_mutations])
        tokens_orig = tokenizer(" ".join(orig_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
        orig_ids = tokens_orig["input_ids"].to(device)
        target_seq_nospaces = "".join([x[1] for x in possible_mutations])
        tokens_target = tokenizer(" ".join(target_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
        target_ids = tokens_target["input_ids"].to(device)



    #mutstate = [0] * len(possible_mutations)
    actions = list(range(len(possible_mutations)))


    df_mutannot = pd.DataFrame(list(zip(actions, pos1, [x[0] for x in possible_mutations], pos2, [x[1] for x in possible_mutations])), columns =['action', 'pos1', 'aa1', 'pos2', 'aa2'])

    print("df_mutannot", df_mutannot)

    #possible_mutations = []
    #
    ## What I need is a dictionary of mapping numeric actions to ex. insertions + deletions
    ## So the examined protein is a concatenation of all the actions, translated to insertions, deletions, or mutations
    #print(target_ids.shape)
    #for key, value in alignment_map.items():
    #    if orig_seq_nospaces[key] != target_seq_nospaces[value]:
    #       print(target_ids[0][value])
    #       possible_mutations.append((key, target_ids[0][value].item())) # Only mutations that would change the sequence
    #       mutannot.append((key, orig_seq_nospaces[key], value, target_seq_nospaces[value]))
    #print("possible_mutations", possible_mutations) 
    #df_mutannot = pd.DataFrame(mutannot, columns =['pos1', 'aa1', 'pos2', 'aa2'])

 
    #env = ProteinMutationEnv(orig_ids.squeeze(), target_ids.squeeze(), possible_mutations, mutstate, cossim_orig_target.item(), model, tokenizer, device)



    env = ProteinMutationEnv(orig_ids.squeeze(), target_ids.squeeze(), possible_mutations,  model, tokenizer, device, int_limits = int_limits, int_ids = int_ids, head_mask = head_mask)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n


    tracker = ActionValueTracker()
    
    # Hyperparameters
    
    learning_rate = 0.1
    gamma = 0.99  # Discount factor
    epsilon = 0.1  # Exploration rate
    epsilon_min = 0.5
    epsilon_decay = 0.995
    update_target_every = 20  # Update target network every 1000 steps

    episodes_done = 0


    hidden_size = 64
    
    
     
    # Q-Network
    q_network = QNetwork(input_size, hidden_size, output_size)
    q_network = q_network.to(device)

    target_q_network = QNetwork(input_size, hidden_size, output_size)
    target_q_network = q_network.to(device)
    update_target_network(q_network, target_q_network)  # Initialize target network


    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_fn_q = nn.MSELoss()
    

    # Replay buffer parameters
    replay_buffer = PrioritizedReplayBuffer(size=100000)
    batch_size = 64 






    
    
    # Training Loop
    # Start with N rounds of random mutations to get cumulative rewards per mut with markov
    # Then M rounds of mutations in order of markov derived q table. 
    for episode in range(num_episodes):
        print("Start episode, ", episode)
        print(tracker.print_cumulative_rewards())
         

      
        counter = 0
        mutstate = env.reset()
        mutstate = torch.tensor(mutstate, dtype=torch.float32).clone().detach().to(device)
        done = False
   
        current_action_space = list(range(env.action_space.n))
        previous_actions = []
        while not done:
          valid_action_found = False
          # Don't need to make the same mutation twice
          while not valid_action_found:            
            #if random.random() < epsilon:
            #    action = env.action_space.sample()  # Explore
            if True:
                 action = random.sample(list(set(current_action_space)), 1)[0] # Explore
                 current_action_space.remove(action)
            else:
                q_values = q_network(state)
                action_values = tracker.get_action_values()
                action = torch.argmax(q_values).item()  # Exploit

            if action not in previous_actions:
                valid_action_found = True
                previous_actions.append(action)


    
          # Take action and observe reward and next state
          if int_fasta_file:
             next_ids, reward, mutstate, best_seqsim, lowest_mse_self, lowest_mse_int, seqsim_history, seqsim_reward, done, _ = env.step(action)
          else:
             next_ids, reward, mutstate, best_seqsim, lowest_mse_self, seqsim_history, seqsim_reward, done, _ = env.step(action)

          next_state = mutstate
          cumulative_rewards_df = tracker.cumulative_rewards_table()
          if len(current_action_space) == 0:
              done = True

          next_state = torch.tensor(next_state, dtype=torch.float32)

          next_state = next_state.clone().detach()  # Added clone detach after warning
          tracker.update(action=action, reward=reward, episode = episode, step = counter)
          counter = counter + 1
          if counter % 20 == 0 or counter == 176:
                  print("mod ", tokenizer.decode(next_ids, skip_special_tokens=True), reward)
                  print("orig",  " ".join(orig_seq_nospaces))
                  print("action", action, possible_mutations[action])
                  print("current best seqsim", best_seqsim)
  
   
          # Define success_metric, here a significant increase in cosine similarity from query to target
          # Don't use success metric. 
          success_metric = float(reward > 0.1) 
   
          # Add experience to the prioritized replay buffer
          experience = (next_ids, action, reward, next_state, done, success_metric)
          replay_buffer.add(experience)

          #print("Episode", episode) 
          if counter ==0:
                  q_network.eval()
                  with torch.no_grad():
                      q_values_check = q_network(torch.tensor(mutstate, dtype=torch.float32).to(device)) 
                  q_network.train()
                  df_mutannot['q'] = q_values_check.clone().detach().cpu() # detach necessary or not?
                  print("Episode", episode) 
                  print(df_mutannot.merge(cumulative_rewards_df, how = "left", on = ["action", "aa1"]).sort_values(by='cumulative_reward', ascending=False).head(100))
 

    ####################################
    #
    # Get the step log    
    step_log = tracker.get_step_log()
    print(step_log)
    # Aggregate cumulative rewards by episode and action
    #total_rewards = step_log.groupby(['action'])['reward'].sum().reset_index(name='total_reward')
    total_rewards = step_log
    # Rank the actions within each episode
    # The sample(frac = 1) randomly sorts the dataframe so that mutations with tied rewards will be done in a random order
    total_rewards['rank'] = total_rewards.sample(frac=1).groupby('episode')['cumulative_reward'].rank(method='first', ascending=False)
    
    # Sort the DataFrame for better readability
    total_rewards.sort_values(by=['episode', 'rank'], inplace=True)
    
    # Display the resulting DataFrame
    print("total rewards", total_rewards)
    
    # Save this DataFrame to a CSV file
    total_rewards.to_csv(f'{outfile_base}_action_rankings_per_total.csv', index=False)

    



    ##### Now see how the markov ranked mutations lead to increasing cosine similarity.
    # Then with MAB, work to refine this rate of increase if possible
    # Add an empty column for similarity scores
    total_rewards['int_attn_mses'] = None
    
    # Get a list of episodes
    episodes = step_log['episode'].unique()
    total_rewards = process_episode_actions(
        total_rewards=total_rewards, 
        episodes=episodes, 
        possible_mutations=possible_mutations, 
        target_ids=target_ids, 
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        int_fasta_file=int_fasta_file  # or True, if you're using int_fasta_file and set the other required parameters accordingly
        # Include other parameters as needed
    )

    total_rewards.to_csv(f'{outfile_base}_action_rankings_total_with_similarity.csv', index=False)
    
    # Filter the DataFrame to only the final episode
    # Or the episode with the largest area under the curve?
    post_markov = total_rewards[total_rewards['episode'] == num_episodes - 1]
    print("POST MARKOV")
    print(post_markov)


    # At this point, save a fasta file of original, with actions applied in order until the target is reached

    # Specify the output FASTA file name

    fasta_dir = f"{output_dir}/fastas"
    os.makedirs(fasta_dir, exist_ok=True) 


    # Iterate over DataFrame rows to make fastas over time with score
    # Ready for batch submission to omegafold
    # Avoid alphafold, because we don't want MSA's helping?

    with open(f"{fasta_dir}/{filename.replace('.fasta', '')}_000.fasta", "w") as f:
        f.write(f">{orig_name}\n{orig_seq_nospaces}\n")
        
    for index, row in post_markov.iterrows():
        base_id = f"{filename.replace('.fasta', '')}_{int(row['rank']):03d}_{row['aa1']}{row['pos1']}_{row['aa2']}{row['pos2']}_{row['similarity_score']}"
        with open(f"{fasta_dir}/{base_id}.fasta", 'w') as f:
             # Write the header and sequence to the file
             f.write(f">{base_id}\n{row['out_seq']}\n")

    with open(f"{fasta_dir}/{filename.replace('.fasta', '')}_TARGET.fasta", "w") as f:
        f.write(f">{target_name}\n{target_seq_nospaces}\n")
 



    current_action_space = post_markov['action'].tolist()
    print("CURRENT_ACTION_SPACE", current_action_space)
    ranks = post_markov['rank'].tolist()

    q_values = [x/num_episodes for x in post_markov['cumulative_reward'].tolist()]

    max_score = max(q_values)    
    scaled_q_values = [100 *x/max_score for x in q_values]   


    q_value_dict = dict(zip(current_action_space, scaled_q_values)) 



    stepwise_seqsims = post_markov['similarity_score'].tolist()
    print("Is ranks in order", ranks, stepwise_seqsims)
    # Example usage
    baseline_aucs = get_auc(stepwise_seqsims)
    print("Baseline AUC values:", baseline_aucs)


    ############## Current best AUC per step
  

    mab_dict = {}
    for action in current_action_space:
        mab_dict[action] = 0
    print("q_value_dict", q_value_dict)
    tracker_mab = MABTracker(q_value_dict = q_value_dict)
    env = ProteinMutationEnv(orig_ids.squeeze(), target_ids.squeeze(), possible_mutations,  model, tokenizer, device, int_limits = int_limits, int_ids = int_ids, head_mask = head_mask, baseline_aucs = baseline_aucs)
    ####################################
    # Now for the Multi Armed Bandit part

    #num_episodes_mab = num_episodes #* 5 
    number_to_advance = 10
    for episode in range(num_episodes_mab):
        print("Start episode, ", episode)
        #print(tracker_mab.print_q_values())
          
        counter = 0
        mutstate = env.reset()
        mutstate = torch.tensor(mutstate, dtype=torch.float32).clone().detach().to(device)
        done = False
   
        #current_action_space = list(range(env.action_space.n))
        previous_actions = []


        
        action_sequence = get_neworder2(current_action_space, number_to_advance = number_to_advance)
        promoted_actions = action_sequence[:number_to_advance]    
        print("action_sequence ", action_sequence)
        print("promoted actions", promoted_actions)
        for action in action_sequence:

          if int_fasta_file:
             next_ids, reward, mutstate, best_seqsim, lowest_mse_self, lowest_mse_int, seqsim_history, seqsim_reward, done, _ = env.step(action)
          else:
             next_ids, reward, mutstate, best_seqsim, lowest_mse_self, seqsim_history, seqsim_reward, done, _ = env.step(action)

          next_state = mutstate
          tracker_mab.update_q_values(action, seqsim_reward, episode = episode, step = counter)


          counter = counter + 1
          #tracker_mab.print_q_values()
          #d ef update_q_values(self, action, reward):

          #if len(current_action_space) == 0:
          #    done = True

        print("orig", cumulative_rewards_df.sort_values(by = "cumulative_reward", ascending= False))
        sorted_q_table = tracker_mab.print_q_table().sort_values(by='q_value', ascending=False)
        print("mab ", sorted_q_table)
        print("prev order", current_action_space)
        current_action_space = sorted_q_table['action'].tolist()
        print("new order ", current_action_space)
        print("seqsim_history", seqsim_history)

         




        print("seqsim_history", seqsim_history)

        print("AUC ANALYSIS")
        print(stepwise_seqsims)
        print(seqsim_history) 
        print(baseline_aucs)
        base_aucs = get_auc(stepwise_seqsims[:-10])
        episode_aucs = get_auc(seqsim_history[10:])
        print(episode_aucs)
        print(len(base_aucs))
        print(len(episode_aucs))

        #
        improvement = episode_aucs[-1] - base_aucs[-1]
        print("IMPROVEMENT?", improvement)        
        if improvement > 0:
              for action in promoted_actions:
                   mab_dict[action] += improvement 
              print("mab_dict", mab_dict)
        print("done")
        
        mab_df = pd.DataFrame.from_dict(mab_dict, orient='index', columns=['improvement'])

        # Reset the index to make 'action' a regular column
        mab_df.reset_index(inplace=True)
        mab_df.columns = ['action', 'improvement']

        # Sort the DataFrame in descending order based on the 'improvement' column
        mab_df_sorted = mab_df.sort_values(by='improvement', ascending=False)

        # Reset the index to start from 0
        mab_df_sorted.reset_index(drop=True, inplace=True)
        print("MAB_DF_SORTED", mab_df_sorted)

    episodes_mab =[]
    if num_episodes_mab > 0:
        step_log_mab = tracker_mab.get_step_log()
    
        print(step_log_mab)
        print(mab_df_sorted)
        # Aggregate cumulative rewards by episode and action
        #total_rewards = step_log.groupby(['action'])['reward'].sum().reset_index(name='total_reward')
        total_rewards_mab = step_log_mab
        # Rank the actions within each episode
        # The sample(frac = 1) randomly sorts the dataframe so that mutations with tied rewards will be done in a random order
        total_rewards_mab['rank'] = total_rewards_mab.sample(frac=1).groupby('episode')['q_value'].rank(method='first', ascending=False)
        
        # Sort the DataFrame for better readability
        total_rewards_mab.sort_values(by=['episode', 'rank'], inplace=True)
        
        # Display the resulting DataFrame
        print("total rewards after bandit", total_rewards_mab)
        
        # Save this DataFrame to a CSV file
        total_rewards_mab.to_csv(f'{outfile_base}_action_rankings_per_total_mab.csv', index=False)
    
    
        total_rewards_mab_steporder = step_log_mab.copy()
        # Rank the actions within each episode
        # The sample(frac = 1) randomly sorts the dataframe so that mutations with tied rewards will be done in a random order
        total_rewards_mab_steporder['rank'] = total_rewards_mab_steporder.sample(frac=1).groupby('episode')['step'].rank(method='first', ascending=True)
        
        # Sort the DataFrame for better readability
        total_rewards_mab_steporder.sort_values(by=['episode', 'rank'], inplace=True)
        
        # Display the resulting DataFrame
        print("total rewards during bandit", total_rewards_mab_steporder)
        
        # Save this DataFrame to a CSV file
        total_rewards_mab_steporder.to_csv(f'{outfile_base}_action_rankings_per_total_mab_steporder.csv', index=False)
    
    
        # Add an empty column for similarity scores
       
      
        total_rewards_mab['similarity_score'] = None
        total_rewards_mab['self_attn_mses'] = None
        total_rewards_mab['int_attn_mses'] = None
     
        # Get a list of episodes
    
        
        episodes_mab = step_log_mab['episode'].unique()
        
    
    
        total_rewards_mab = process_episode_actions(
            total_rewards=total_rewards_mab, 
            episodes=episodes_mab, 
            possible_mutations=possible_mutations, 
            target_ids=target_ids, 
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            int_fasta_file=int_fasta_file  # or True, if you're using int_fasta_file and set the other required parameters accordingly
            # Include other parameters as needed
        )
     
        # Display the resulting DataFrame
        print("total rewards_mab", total_rewards_mab)
    
        total_rewards_mab_steporder = process_episode_actions(
            total_rewards=total_rewards_mab_steporder, 
            episodes=episodes_mab, 
            possible_mutations=possible_mutations, 
            target_ids=target_ids, 
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            int_fasta_file=int_fasta_file  # or True, if you're using int_fasta_file and set the other required parameters accordingly
            # Include other parameters as needed
        )
     
        total_rewards_mab.to_csv(f'{outfile_base}_action_rankings_total_mab_with_similarity.csv', index=False)
       
    

 
    if "this" == True: 
          # Add experience to the prioritized replay buffer
          experience = (next_ids, action, reward, next_state, done, success_metric)
          replay_buffer.add(experience)





  
          # Here is where we track successes? 
  
  
          # Check if enough samples are available in the buffer
              
          if len(replay_buffer.buffer) >   batch_size:
              
                  # Sample a batch from the replay buffer
              
                  batch = replay_buffer.sample(batch_size)
              
                  ids, actions, rewards, next_states, dones, _ = zip(*batch)
              
              
              
                  # My states should be the action boolean vector...
                  # Convert to tensors
                  actions = torch.tensor(actions, dtype=torch.long).to(device)
                  rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            
                  #padded_states = [torch.nn.functional.pad(tensor, (0, max_size - tensor.size(0))) for tensor in next_states]
            
                  #stacked_states = torch.stack(padded_states).to(device)                  

                  #print(next_states)



                  next_states = torch.stack(next_states).to(device)
                  dones = torch.tensor(dones, dtype=torch.float32).to(device)
                  #print(f"state_shape {states.shape}") 
                  # Compute Q values for current states
                  q_values = q_network(next_states)
                  #print(f"q_values shape {q_values.shape}")            
  
                  # OK I'm currently not using a target network. Should we?
                  # Compute Q values for next states
                  next_q_values = target_q_network(next_states)
             
  
   
                  # Select the Q value for the action taken, since we only compute loss on actions that were taken
                  q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                  #print(f"q_values shape {q_values.shape}")            
             
                  # Compute the expected Q values
                  q_targets_next = next_q_values.max(1)[0]
                  q_targets = rewards + (gamma * q_targets_next * (1 - dones))
              
   
                  # Update Q-Network
                  optimizer.zero_grad()
                  loss_q = loss_fn_q(q_values, q_targets.detach())
                  loss_q.backward()
                  optimizer.step()
  
      # Update target network at the end of each episode or after certain steps
  
          if episode % update_target_every == 0:
              update_target_network(q_network, target_q_network)
  
          #print("Episode", episode) 
          if counter ==0:
                  q_network.eval()
                  with torch.no_grad():
                      q_values_check = q_network(torch.tensor(mutstate, dtype=torch.float32).to(device)) 
                  q_network.train()
                  df_mutannot['q'] = q_values_check.clone().detach().cpu() # detach necessary or not?
                  print("Episode", episode) 
                  print(df_mutannot.merge(cumulative_rewards_df, how = "left", on = ["action", "aa1"]).sort_values(by='cumulative_reward', ascending=False).head(100))
   
          counter = counter + 1    
      
      
          # Update epsilon
      
          epsilon = max(epsilon_min, epsilon_decay * epsilon)
          # Logging
     
            
          #if episode % 1 == 0:
          #    print(f"Episode: {episode}, Loss: {loss_q.item()}, Epsilon: {epsilon}")
            
    
    
    # Close the environment



      
   
    


    # Now random order
    random_rewards = total_rewards.copy()
    random_rewards['similarity_score'] = None
    random_rewards['self_attn_mses'] = None
    random_rewards['int_attn_mses'] = None

    # What about by the similarity score using the random steps?
    random_rewards= random_rewards.sort_values(by='step')

    random_rewards = process_episode_actions(
        total_rewards=random_rewards, 
        episodes=episodes, 
        possible_mutations=possible_mutations, 
        target_ids=target_ids, 
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        int_fasta_file=int_fasta_file  # or True, if you're using int_fasta_file and set the other required parameters accordingly
        # Include other parameters as needed
    )
 
    random_rewards['rank'] = random_rewards['step'] 


    ############
    # Set up plot
    # Set up a figure with two subplots (side by side)
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))  # 2 rows, 1 column
    
    
    
    # Common settings for both plots
    
    cmap_viridis = plt.cm.viridis
    
    num_episodes = len(episodes)
    
    colors_viridis = cmap_viridis(np.linspace(0, 1, num_episodes))
    
    
    
    cmap_magma = plt.cm.magma
    
    num_episodes_mab = len(episodes_mab)
    
    colors_magma = cmap_magma(np.linspace(0, 1, num_episodes_mab))
    
    
    
    # Plot the grey group on both subplots
    
    for episode in episodes:
    
        episode_data_random = random_rewards[random_rewards['episode'] == episode]
    
        # Plot on the first subplot
    
        axs[0].plot(episode_data_random['rank'], episode_data_random['similarity_score'], color=(0.8, 0.8, 0.8), zorder=1)
    
        # Plot on the second subplot
    
        axs[1].plot(episode_data_random['rank'], episode_data_random['similarity_score'], color=(0.8, 0.8, 0.8), zorder=1)
    
    
    
    # Plot the viridis group on the first subplot
    
    for idx, episode in enumerate(episodes):
    
        episode_data = total_rewards[total_rewards['episode'] == episode]
    
        axs[0].plot(episode_data['rank'], episode_data['similarity_score'], color=colors_viridis[idx], zorder=2)
    


    
    
    # Plot the magma group on the second subplot
    
    for idx, episode in enumerate(episodes_mab):
    
        episode_data = total_rewards_mab[total_rewards_mab['episode'] == episode]
    
        axs[1].plot(episode_data['rank'], episode_data['similarity_score'], color=colors_magma[idx], zorder=3)



    # Plot the magma group on the 3rd subplot
    
    for idx, episode in enumerate(episodes_mab):
    
        episode_data = total_rewards_mab_steporder[total_rewards_mab_steporder['episode'] == episode]
        episode_data_baseline = total_rewards[total_rewards['episode'] == episodes[-1]]
           

        axs[2].plot(episode_data['rank'], episode_data['similarity_score'], color=colors_magma[idx], zorder=3)
        axs[2].plot(episode_data_baseline['rank'] + 10, episode_data_baseline['similarity_score'], color="red", zorder=4)
    
    
 
 
    
    # Set labels for both subplots
    
    axs[0].set_xlabel('Rank')
    
    axs[0].set_ylabel('Similarity score')
    
    axs[0].set_title('Similarity vs Ranked mutation per Episode: Markov')
    
    
    
    axs[1].set_xlabel('Rank')
    
    axs[1].set_ylabel('Similarity score')
    
    axs[1].set_title('Similarity vs. Ranked mutation per Episode: MAB')


    axs[2].set_xlabel('Rank')
    
    axs[2].set_ylabel('Similarity score')
    
    axs[2].set_title('Similarity vs. Action order per Episode: MAB')
     
    plt.tight_layout()  # Adjust layout to not overlap
    # Get the current x-axis limits

    x_min, x_max = plt.xlim()

    # Calculate the range for ticks
    tick_range = np.arange(np.ceil(x_min / 10) * 10, np.floor(x_max / 10) * 10 + 1, 10)

    # Set ticks on the x-axis
    plt.xticks(tick_range)

    plt.show()
    

    plt.savefig(f'{outfile_base}_similarity_score_plot.png')
    plt.savefig(f'{outfile_base}_similarity_score_plot.pdf')
    plt.close()
    
 




    plt.figure(figsize=(10, 6))
    

    for idx, episode in enumerate(episodes):
    
        # Filter the data for the current episode
        episode_data_random = random_rewards[random_rewards['episode'] == episode]
     
        # Plot the data
        plt.plot(episode_data_random['rank'], episode_data_random['self_attn_mses'], color = (0.8, 0.8, 0.8), zorder = 1)
        # just a few background lines
        if episode >=10:
            continue     

    
    for idx, episode in enumerate(episodes):
    
        # Filter the data for the current episode
        episode_data = total_rewards[total_rewards['episode'] == episode]
    
        
    
        # Plot the data
        plt.plot(episode_data['rank'], episode_data['self_attn_mses'], label=episode, color = colors[idx], zorder = 2)
    
    
    
    plt.xlabel('Rank')
    plt.ylabel('Self attn mses')
    plt.title('Ranked mutation vs. Self attention MSES for each Episode')
    plt.show()
    plt.savefig(f'{outfile_base}_attn_mse_plot.png')
    plt.savefig(f'{outfile_base}_attn_mse_score_plot.pdf')
    plt.close()
 
    #    for idx, episode in enumerate(episodes):
    #            episode_data = total_rewards[total_rewards['episode'] == episode]
    #    
    #        
    #    
    #            # Plot the data
    #            plt.plot(episode_data['rank'], episode_data['int_attn_mses'], color = colors[idx])
    #        
    #        
    #        
    #            plt.xlabel('Rank')
    #            plt.ylabel('MSE')
    #            plt.title('MSE vs Ranked mutation for each Episode')
    #            plt.show()
    #            plt.savefig(f'{fasta_file}_intattn_mse_score_plot.png')
    #             
    #    
    #
    #        plt.figure(figsize=(10, 6))
    #    
    #        # This is not a really useful plot... 
    #        for idx, episode in enumerate(episodes):
    #    
    #            # Filter the data for the current episode
    #            episode_data = total_rewards[total_rewards['episode'] == episode]
    #    
    #        
    #    
    #            # Plot the data
    #            plt.plot(episode_data['int_attn_mses'], episode_data['similarity_score'], color = colors[idx], label=episode)
    #        
    #        
    #        
    #            plt.xlabel('Rank')
    #            plt.ylabel('MSE')
    #            plt.title('Int  attn_score vs. similarity_score for each Episode')
    #            plt.show()
    #            plt.savefig(f'{fasta_file}_mse_sim_score_plot.png')
    # 
    #
    #
    # 
    #    #############################
    #
    #
    #    # Now make some plots of the random distribution
    #    # Set up plot
    #    num_episodes = len(episodes)
    #    cmap = plt.cm.viridis  # You can change 'viridis' to any other colormap
    #    colors = cmap(np.linspace(0, 1, num_episodes))
    #
    #    plt.figure(figsize=(10, 6))
    #        
    #    for idx, episode in enumerate(episodes):
    #    
    #        # Filter the data for the current episode
    #        episode_data = random_rewards[random_rewards['episode'] == episode]
    #    
    #        # Plot the data
    #        plt.plot(episode_data['step'], episode_data['similarity_score'], label=episode, color = colors[idx])
    #
    #    plt.xlabel('Random step')
    #    plt.ylabel('Similarity score')
    #    plt.title('Similarity vs Random mutation for each Episode')
    #    plt.show()
    #    plt.savefig(f'{fasta_file}_similarity_score_plot_random.png')
    #
    #    
    #    random_rewards.to_csv(f'{fasta_file}_action_rankings_total_with_similarity_random.csv', index=False)
    #    
    #
    #    for idx, episode in enumerate(episodes):
    #    
    #        # Filter the data for the current episode
    #        episode_data = random_rewards[random_rewards['episode'] == episode]
    #    
    #        # Plot the data
    #        plt.plot(episode_data['step'], episode_data['self_attn_mses'], label=episode, color = colors[idx])
    #
    #    plt.xlabel('Random step')
    #    plt.ylabel('Self attn mse')
    #    plt.title('Random mutation vs. Self attn MSE for each Episode')
    #    plt.show()
    #    plt.savefig(f'{fasta_file}_self_attn_mse_plot_random.png')
    #
    #    
    #    
    #
    #
    #


    
    env.close()
    exit(1)
    
