import argparse
from Bio import SeqIO, AlignIO
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel, AutoConfig
import gc
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import SGD
from sklearn.metrics.pairwise import cosine_similarity
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile
import subprocess
from io import StringIO
from Bio.Align import MultipleSeqAlignment

import os
from datetime import datetime
import math

import psutil
import pandas as pd
pd.set_option('display.max_rows', 2000)
import random

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt
#import seaborn as sns


from gymnasium import spaces

from time import time

import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader


   # For a window_size of 5:
   #Original: [A->B->C->D->E->F->G->H]
   #Window 1: [A->B->C->D->E]->F->G->H
   #Window 2: A->[B->C->D->E->F]->G->H
   #Window 3: A->B->[C->D->E->F->G]->H
 

   #After Change: E -> A -> B -> C -> D  -> F -> G -> H
   #Window 2:    E -> [A -> B -> C -> D -> F] -> G -> H


   #For each window:
   #1. Generate random permutations
   #2. Evaluate each permutation
   #3. Keep top beam_width best ones
   #4. Continue with best candidates

def print_memory_stats():
    """Helper to print memory usage"""
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f}MB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 / 1024:.2f}MB")
            print(f"GPU {i} Memory Reserved: {torch.cuda.memory_reserved(i) / 1024 / 1024:.2f}MB")
            print(f"GPU {i} Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024 / 1024:.2f}MB")

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        # Clear the cache
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Force garbage collection
        gc.collect()
        
        # If still having issues, can try to explicitly free memory
        if hasattr(torch.cuda, 'memory_summary'):
            torch.cuda.memory_summary(device=None, abbreviated=False)
            
        # Can also try releasing all unoccupied cached memory
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

########################
# ENVIRONMENT
########################
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
        baseline_aucs (list, optional): A list of baseline AUCs for evaluation. Defaults to an empty list.

    The environment defines an action space of possible mutations and an observation space representing the
    current state of mutations applied to the original sequence. It uses the transformer model to evaluate
    the impact of mutations and provides rewards based on the similarity of the mutated sequence to the target
    sequence and other criteria.
    """

    def __init__(self, orig_ids, target_ids, possible_mutations,  model, tokenizer, device, baseline_aucs = [], baseline_reward=False, starting_seqsim=None):

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
        self.baseline_aucs = baseline_aucs
        self.baseline_reward = baseline_reward
        self.starting_seqsim = starting_seqsim
        print("possible_mutations", self.possible_mutations)

        self.seqsim_history = []

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
            self.orig_mean_embedding = get_representation(self.model, self.orig_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = False)
            self.target_mean_embedding = get_representation(self.model, self.target_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = False)
            #print(orig_mean_embedding)
            #print(target_mean_embedding)
            self.starting_seqsim = F.cosine_similarity(self.orig_mean_embedding.to(device), self.target_mean_embedding.to(device)).item()
            self.best_seqsim = self.starting_seqsim


 

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
            current_mean_embedding = get_representation(self.model, self.current_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = False)
            #print("Current embedding shape:", current_mean_embedding.shape)  # Add debug print
        seqsim = F.cosine_similarity(current_mean_embedding.to(device), self.target_mean_embedding.to(device)).item()
        print(f"Action {action}: {possible_mutations[action]} -> New similarity: {seqsim}")  # Add debug print
        
        reward_ss = seqsim - self.best_seqsim

        # Only get a reward for improving on the cosine similarity
        
        # Worse
        if reward_ss <= 0:
            reward_ss = 0

        # Better
        else:
            self.best_seqsim = seqsim
      

        reward =  reward_ss

        self.seqsim_history.append(seqsim)
        self.latest_reward = reward # not using this  
        self.mutation_counter += 1  # Increment mutation counter

        if self.baseline_aucs:
           auc_reward = get_auc(self.seqsim_history)[-1] - self.baseline_aucs[len(self.seqsim_history) - 1]
           #print("AUC reward", auc_reward)
           if auc_reward > 0:
              auc_reward =  auc_reward
           else:
              #reward = 0
              auc_reward = 0

        else:
            auc_reward = []
        return self.current_ids, reward, self.mutstate, self.best_seqsim, self.seqsim_history, auc_reward, {}



    def reset(self):
        self.mutstate = [0] * len(self.possible_mutations) 
        self.current_ids = self.orig_ids.clone().detach()
        self.mutation_counter = 0  # Reset the mutation counter
        self.latest_reward = 0 
        self.seqsim_history = []

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

    def get_initial_state(self):
        """Returns a fresh initial mutation state by resetting first"""
        self.reset()
        return self.mutstate


########################
# TRACKING
########################
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

    def __init__(self, starting_seqsim):
        self.step_log = []
        self.previous_similarity = None
        self.starting_seqsim = starting_seqsim
        
    def update(self, action, episode, step, current_similarity):
        # If this is the first update of the episode, store starting similarity
        if self.previous_similarity is None:
            self.previous_similarity = self.starting_seqsim
        # Convert tensor/array to scalar if needed
        if hasattr(current_similarity, 'item'):
            current_similarity = current_similarity.item()
        
        # Calculate improvement over previous state
        improvement = 0.0
        if self.previous_similarity is not None:
            improvement = max(0, current_similarity - self.previous_similarity)
        
        # Update previous similarity for next step
        self.previous_similarity = current_similarity
        
        # Log this step with all relevant information
        log_entry = {
            'episode': episode,
            'step': step,
            'action': action,
            'aa1': possible_mutations[action][0],
            'pos1': pos1[action],
            'aa2': possible_mutations[action][1],
            'pos2': pos2[action],
            'similarity_score': current_similarity,
            'improvement': improvement,
        }
        
        self.step_log.append(log_entry)

    def reset_episode(self, starting_similarity):
        """Reset episode-specific tracking"""
        self.previous_similarity = None
        self.starting_similarity = starting_similarity

    def get_step_log(self):
        return pd.DataFrame(self.step_log)

    def calculate_cumulative_rewards(self):
        """Calculate cumulative rewards when needed"""
        cumulative_rewards = {}
        action_counts = {}
        
        # Process the step log to calculate rewards
        for step in self.step_log:
            action = step['action']
            improvement = step['improvement']
            
            if action not in cumulative_rewards:
                cumulative_rewards[action] = 0
                action_counts[action] = 0
            
            cumulative_rewards[action] += improvement
            action_counts[action] += 1
        
        # Calculate average rewards
        action_values = {
            action: cumulative_rewards[action] / action_counts[action]
            for action in cumulative_rewards
        }
        
        return action_values, pd.DataFrame({
            'action': list(cumulative_rewards.keys()),
            'aa1': [possible_mutations[x][0] for x in cumulative_rewards.keys()],
            'cumulative_reward': list(cumulative_rewards.values())
        })



  


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


########################
# RESULTS PROCESSING
########################


def apply_actions_in_order(actions, possible_mutations, target_ids, model, tokenizer, device, only_final=False):
    """Apply mutations in order and get sequence similarities.
    
    Args:
        actions: List of mutation indices to apply
        possible_mutations: List of possible mutations
        target_ids: Target sequence IDs
        model: The model
        tokenizer: The tokenizer
        device: The computation device
        only_final: If True, only compute similarity for final sequence (default: False)
    
    Returns:
        tuple: (similarities, sequences)
    """
    current_mutstate = [0] * len(possible_mutations)
    sequences = []
    similarities = []
    
    for i, action in enumerate(actions):
        # Apply mutation
        current_mutstate[action] = 1
        sequence, _ = create_substitution_mask(current_mutstate, possible_mutations)
        sequences.append(" ".join(sequence))
        
        # Only evaluate similarity if it's the final sequence or if we want all similarities
        if not only_final or i == len(actions) - 1:
            tokens = tokenizer(sequences[-1], return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                current_embedding = get_representation(model, 
                                                     tokens["input_ids"], 
                                                     "t5",  # Note: Should pass model_type instead of hardcoding
                                                     layers=[-1], 
                                                     output_attentions=False)
                similarity = F.cosine_similarity(current_embedding, target_mean_embedding)
                similarities.append(float(similarity.cpu().numpy()))
        else:
            # If not evaluating, copy previous similarity
            similarities.append(similarities[-1] if similarities else 0.0)
    
    return similarities, sequences

def process_episode_actions(total_rewards, episodes, possible_mutations, target_ids, model, tokenizer, device):
    for episode in episodes:
        # Get actions for this episode sorted by rank
        episode_actions = total_rewards[total_rewards['episode'] == episode]
        actions_list = episode_actions['action'].tolist()
        # Apply actions in order and update DataFrame
        similarity_score, out_seqs = apply_actions_in_order(
                actions_list, possible_mutations, target_ids, model, tokenizer, device 
            )
        total_rewards.loc[total_rewards['episode'] == episode, 'similarity_score'] = similarity_score
        total_rewards.loc[total_rewards['episode'] == episode, 'out_seq'] = out_seqs
   
    return total_rewards


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


########################
# CONFIGURATION
########################

def parse_arguments():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Optimize protein sequence embeddings with a model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the pretrained model.")

    parser.add_argument("-f", "--fasta_file", type=str, required=True, help="Path to the FASTA file with two sequences: the query and the target.")




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
 
    # New argument for loading previous results
    parser.add_argument("--step_log", type=str, help="Path to previously saved step_log.csv file. If provided, skips trials and goes directly to pathway analysis.")
    
    # New arguments for pathway analysis
    parser.add_argument("-n", "--n_subsamples", type=int, default=20,
                       help="Number of random subsamples to analyze for pathway analysis (default: 20)")
    parser.add_argument("-s", "--subsample_size", type=int, default=100,
                       help="Number of episodes per subsample for pathway analysis (default: 100)")

    # Add new argument for loading previous pathway analysis
    parser.add_argument("--pathway_file", type=str, 
                       help="Path to previously saved mutation_pathway_analysis.csv file. If provided, skips trials and pathway analysis.")
    

    # Add optimization method selection
    parser.add_argument("--optimization-method", 
                       choices=['original', 'beam', 'both'],
                       default='both',
                       help="Choose optimization method: 'original' for window-based, 'beam' for beam search, or 'both' (default)")
    
    # Original optimization parameters
    parser.add_argument("--window-size", 
                       type=int, 
                       default=2,
                       help="Window size for original optimization (default: 2)")
    parser.add_argument("--n-best-paths", 
                       type=int, 
                       default=3,
                       help="Number of best paths to consider for original optimization (default: 3)")
    
    # Beam search parameters
    parser.add_argument("--beam-width", 
                       type=int, 
                       default=5,
                       help="Number of paths to keep at each level in beam search (default: 5)")
    parser.add_argument("--look-ahead", 
                       type=int, 
                       default=5,
                       help="Number of mutations to look ahead in beam search (default: 5)")

    parser.add_argument('--beam-iterations', 
                      type=int,
                      default=3,
                      help='Number of beam search iterations to perform (default: 3)')
        
    # Add baseline reward flag
    parser.add_argument('--baseline-reward',
                       action='store_true',
                       help='Only give positive rewards for improvements above the original sequence similarity')

    # Add max_steps argument
    parser.add_argument('--max_steps', 
                       type=int,
                       default=None,
                       help='Maximum number of mutation steps to optimize. If not specified, optimizes full path')
    
    # Add new argument for reference sequences
    parser.add_argument("-ref", "--reference_fasta", 
                       type=str, 
                       help="Path to FASTA file containing reference sequences to plot")
    
    return parser.parse_args()

########################
# SEQUENCE UTILITIES
########################
def generate_random_protein(n):
    '''
    Given a length n, generate a protein of that length
    '''
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    protein_sequence = ''.join(random.choice(amino_acids) for _ in range(n))
    return protein_sequence
 

def do_align_new(seqrecordlist, filename):
    """
    Align sequences using clustalo via subprocess
    """
    import subprocess
    
    # Write sequences to temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as temp_fasta:
        SeqIO.write(seqrecordlist, temp_fasta, "fasta")
        temp_fasta_name = temp_fasta.name

    # Define output alignment file
    output_file = f"{filename}.output.aln"
    
    # Build clustalo command
    cmd = [
        "clustalo",                  # Command name
        "-i", temp_fasta_name,       # Input file
        "-o", output_file,          # Output file
        "--force",                  # Overwrite existing files
        "--verbose"                 # Print progress
    ]

    # Run alignment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Read the resulting alignment
        alignment = AlignIO.read(output_file, "fasta")
        return alignment
        
    except subprocess.CalledProcessError as e:
        print(f"Alignment failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise
    finally:
        # Clean up temp file
        os.unlink(temp_fasta_name)



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




########################
# MODEL UTILITIES
########################
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


def get_representation(model, input_ids, model_type, layers, mask = None, output_attentions = False, remove_0var_attns = False):
    with torch.no_grad():
        model_output = model.encoder(input_ids=input_ids, output_attentions = output_attentions)
        if output_attentions == True:
            attns = model_output[-1]
        aa_embeddings, aa_shape = retrieve_aa_embeddings(model_output, model_type = model_type, layers = layers)
        aa_embeddings = aa_embeddings
        if mask is not None:
            bool_mask = torch.tensor(mask, dtype=torch.bool)
            masked_embeddings = aa_embeddings[:, ~bool_mask]
            sequence_representation = torch.mean(masked_embeddings, dim = 1)
        else:
            sequence_representation = torch.mean(aa_embeddings, dim = 1)


    if output_attentions == True:
        return(sequence_representation, attns)
    else:
        return(sequence_representation)



########################
# ENVIRONMENT
########################

def update_target_network(primary_network, target_network):

    target_network.load_state_dict(primary_network.state_dict())




def process_and_save_results(step_log, output_base, possible_mutations, target_ids, model, tokenizer, device):
    """Process results and save to files."""
    total_rewards = step_log
    total_rewards['rank'] = total_rewards.sample(frac=1).groupby('episode')['cumulative_reward'].rank(method='first', ascending=False)
    total_rewards.sort_values(by=['episode', 'rank'], inplace=True)
    
    # Save initial rankings
    total_rewards.to_csv(f'{output_base}_action_rankings_per_total.csv', index=False)
    
    # Process episodes and save with similarity scores
    episodes = step_log['episode'].unique()
    total_rewards = process_episode_actions(
        total_rewards=total_rewards,
        episodes=episodes,
        possible_mutations=possible_mutations,
        target_ids=target_ids,
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    total_rewards.to_csv(f'{output_base}_action_rankings_total_with_similarity.csv', index=False)
    return total_rewards, episodes

def run_mab_episodes(env, num_episodes_mab, current_action_space, baseline_aucs, possible_mutations, tracker_mab, number_to_advance=10):
    """Run Multi-Armed Bandit episodes."""
    mab_dict = {action: 0 for action in current_action_space}
    
    for episode in range(num_episodes_mab):
        print("Start episode, ", episode)
        counter = 0
        mutstate = env.reset()
        mutstate = torch.tensor(mutstate, dtype=torch.float32).clone().detach().to(device)
        done = False
        
        action_sequence = get_neworder2(current_action_space, number_to_advance=number_to_advance)
        promoted_actions = action_sequence[:number_to_advance]
        
        for action in action_sequence:
            next_ids, reward, mutstate, best_seqsim, seqsim_history, seqsim_reward, _ = env.step(action)
            next_state = mutstate
            tracker_mab.update_q_values(action, seqsim_reward, episode=episode, step=counter)
            counter += 1
            
        # Update MAB dictionary based on improvements
        base_aucs = get_auc(stepwise_seqsims[:-10])
        episode_aucs = get_auc(seqsim_history[10:])
        improvement = episode_aucs[-1] - base_aucs[-1]
        
        if improvement > 0:
            for action in promoted_actions:
                mab_dict[action] += improvement
                
    return tracker_mab.get_step_log(), mab_dict

def plot_similarity_scores(total_rewards, random_rewards, total_rewards_mab, total_rewards_mab_steporder, 
                         episodes, episodes_mab, output_base):
    """Generate and save similarity score plots."""
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    
    cmap_viridis = plt.cm.viridis
    cmap_magma = plt.cm.magma
    colors_viridis = cmap_viridis(np.linspace(0, 1, len(episodes)))
    colors_magma = cmap_magma(np.linspace(0, 1, len(episodes_mab)))
    
    # Plot random episodes background
    for episode in episodes:
        episode_data_random = random_rewards[random_rewards['episode'] == episode]
        axs[0].plot(episode_data_random['rank'], episode_data_random['similarity_score'], 
                   color=(0.8, 0.8, 0.8), zorder=1)
        axs[1].plot(episode_data_random['rank'], episode_data_random['similarity_score'], 
                   color=(0.8, 0.8, 0.8), zorder=1)
    
    # Plot Markov episodes
    for idx, episode in enumerate(episodes):
        episode_data = total_rewards[total_rewards['episode'] == episode]
        axs[0].plot(episode_data['rank'], episode_data['similarity_score'], 
                   color=colors_viridis[idx], zorder=2)
    
    # Plot MAB episodes
    for idx, episode in enumerate(episodes_mab):
        episode_data = total_rewards_mab[total_rewards_mab['episode'] == episode]
        axs[1].plot(episode_data['rank'], episode_data['similarity_score'], 
                   color=colors_magma[idx], zorder=3)
        
        episode_data = total_rewards_mab_steporder[total_rewards_mab_steporder['episode'] == episode]
        episode_data_baseline = total_rewards[total_rewards['episode'] == episodes[-1]]
        axs[2].plot(episode_data['rank'], episode_data['similarity_score'], 
                   color=colors_magma[idx], zorder=3)
        axs[2].plot(episode_data_baseline['rank'] + 10, episode_data_baseline['similarity_score'], 
                   color="red", zorder=4)
    
    # Set labels and titles
    for ax, title in zip(axs, ['Similarity vs Ranked mutation per Episode: Markov',
                              'Similarity vs. Ranked mutation per Episode: MAB',
                              'Similarity vs. Action order per Episode: MAB']):
        ax.set_xlabel('Rank')
        ax.set_ylabel('Similarity score')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(f'{output_base}_similarity_score_plot.png')
    plt.savefig(f'{output_base}_similarity_score_plot.pdf')
    plt.close()

def save_fasta_outputs(output_dir, filename, orig_name, target_name, orig_seq_nospaces, 
                      target_seq_nospaces, post_markov):
    """Save FASTA outputs for each mutation step."""
    fasta_dir = f"{output_dir}/fastas"
    os.makedirs(fasta_dir, exist_ok=True)
    
    # Save original sequence
    with open(f"{fasta_dir}/{filename.replace('.fasta', '')}_000.fasta", "w") as f:
        f.write(f">{orig_name}\n{orig_seq_nospaces}\n")
    
    # Save intermediate sequences
    for index, row in post_markov.iterrows():
        base_id = f"{filename.replace('.fasta', '')}_{int(row['rank']):03d}_{row['aa1']}{row['pos1']}_{row['aa2']}{row['pos2']}_{row['similarity_score']}"
        with open(f"{fasta_dir}/{base_id}.fasta", 'w') as f:
            f.write(f">{base_id}\n{row['out_seq']}\n")
    
    # Save target sequence
    with open(f"{fasta_dir}/{filename.replace('.fasta', '')}_TARGET.fasta", "w") as f:
        f.write(f">{target_name}\n{target_seq_nospaces}\n")

def setup_model_and_tokenizer(model_path):
    """Initialize the model and tokenizer."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    model = T5ForConditionalGeneration.from_pretrained(model_path, output_hidden_states=True).eval()
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model_config = AutoConfig.from_pretrained(model_path)
    model_type = model_config.model_type
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_memory_stats()
    model = model.half()
    clear_gpu_memory()
    model = model.to(device)
    
    # Ensure model parameters are not updated
    for param in model.parameters():
        param.requires_grad = False
        
    return model, tokenizer, model_type, device

def setup_sequences(args, outfile_base):
    """Setup and process input sequences."""
    if args.fasta_file:
        with open(args.fasta_file, "r") as handle:
            sequences = list(SeqIO.parse(handle, "fasta"))
        
        path, filename = os.path.split(args.fasta_file)
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
        output_dir = os.path.join(path, f"{filename}_{date_str}_ep{args.episodes}_epmab{args.episodes_mab}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input parameters
        params_file = os.path.join(output_dir, "input_params.txt")
        with open(params_file, "w") as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        
        if args.random_start:
            target_seqrecord = sequences[0]
            target_seq_nospaces = str(target_seqrecord.seq)
            target_name = target_seqrecord.id
            
            orig_seqrecord = SeqRecord(Seq(generate_random_protein(len(target_seqrecord.seq))), id="random_seq")
            orig_seq_nospaces = str(orig_seqrecord.seq)
            orig_name = orig_seqrecord.id
            
            alignment = AlignIO.MultipleSeqAlignment([target_seqrecord, orig_seqrecord])
            AlignIO.write(alignment, f"{outfile_base}.randstart.aln", "fasta")
        else:
            orig_seqrecord = sequences[0]
            target_seqrecord = sequences[1]
            orig_name = orig_seqrecord.id
            target_name = target_seqrecord.id
            orig_seq_nospaces = str(orig_seqrecord.seq)
            target_seq_nospaces = str(target_seqrecord.seq)
            
            if not args.aln_file:
                alignment = do_align_new([target_seqrecord, orig_seqrecord], outfile_base)
            else:
                alignment = AlignIO.read(args.aln_file, "fasta")
                orig_seqrecord = alignment[0]
                target_seqrecord = alignment[1]
                orig_name = orig_seqrecord.id
                target_name = target_seqrecord.id
                alignment = AlignIO.MultipleSeqAlignment([target_seqrecord, orig_seqrecord])
        
        if args.xgap:
            target_seq_nospaces = str(alignment[0].seq).replace("-", "X")
            orig_seq_nospaces = str(alignment[1].seq).replace("-", "X")
            
        AlignIO.write(alignment, f"{outfile_base}.aln", "fasta")
        
        return (orig_seqrecord, target_seqrecord, orig_name, target_name, 
                orig_seq_nospaces, target_seq_nospaces, alignment, output_dir, filename)

def analyze_mutation_pathways(total_rewards, possible_mutations, target_ids, model, tokenizer, device, n_subsamples=20, subsample_size=50):
    """
    Analyzes multiple subsamples to find mutation pathways based on cumulative seqsim improvements.
    
    Args:
        total_rewards (pd.DataFrame): Results from all episodes containing episode, action, reward, similarity_score
        possible_mutations (list): List of possible mutations
        target_ids (tensor): Target sequence IDs
        model: The transformer model
        tokenizer: The tokenizer
        device: The device (cuda/cpu)
        n_subsamples (int): Number of random subsamples to analyze (default: 20)
        subsample_size (int): Number of episodes per subsample (default: 50)
    """
    episodes = total_rewards['episode'].unique()
    pathway_results = []
    
    for i in range(n_subsamples):
        print("Subsample", i)
        # 1. Randomly sample a subset of episodes
        sampled_episodes = np.random.choice(episodes, size=subsample_size, replace=False)
        subsample = total_rewards[total_rewards['episode'].isin(sampled_episodes)]
        
        # 2. Calculate total improvement for each mutation across sampled episodes
        mutation_improvements = subsample.groupby('action')['improvement'].sum().reset_index()
        
        # 3. Rank mutations by their total improvement
        ranked_mutations = mutation_improvements.sort_values('improvement', ascending=False)
        mutation_order = ranked_mutations['action'].tolist()
        mutation_strings = [format_mutation(possible_mutations[idx], pos1, pos2) 
                          for idx in mutation_order]
        

        # 4. Apply mutations in ranked order and get actual seqsim values
        seqsim_values, sequences = apply_actions_in_order(
            mutation_order, 
            possible_mutations, 
            target_ids, 
            model, 
            tokenizer, 
            device
        )
        
        # 5. Calculate AUC for this ordering
        cumulative_auc = get_auc(seqsim_values)
        
        # 6. Store results for this subsample
        pathway_results.append({
            'subsample_id': i,
            'mutation_order': mutation_order,
            'mutation_strings': mutation_strings,  # Add formatted strings
            #'mutation_improvements': ranked_mutations['improvement'].tolist(),
            'seqsim_progression': seqsim_values,
            'sequences': sequences,
            'cumulative_auc': cumulative_auc,
            'final_auc': cumulative_auc[-1],
            'final_seqsim': seqsim_values[-1]
        })
        #Print progress with mutation strings
        print(f"Pathway {i+1}")
        print(f"Final AUC: {cumulative_auc[-1]}")
        print("Mutation order:")
        for j, mut_str in enumerate(mutation_strings[:10]):
            print(f"  {j+1}. {mut_str}")
        if len(mutation_strings) > 10:
            print("  ...")
        print() 
    return pd.DataFrame(pathway_results)

def create_consensus_graph(pathway_df, possible_mutations, pos1, min_frequency=0.3):
    """
    Create and visualize a consensus directed graph from multiple mutation pathways.
    Also saves network data in Cytoscape-compatible format.
    
    Args:
        pathway_df: DataFrame containing different mutation pathways
        possible_mutations: List of possible mutations
        pos1: List of positions corresponding to mutations
        min_frequency: Minimum frequency to include an edge (0-1)
    """
    import networkx as nx
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Track both edge counts and AUC sums
    edge_counts = {}
    edge_aucs = {}
    total_pathways = len(pathway_df)
    
    # Analyze each pathway
    for _, pathway in pathway_df.iterrows():
        mutation_order = pathway['mutation_order']
        pathway_auc = pathway['final_auc']
        
        # Look at consecutive mutations
        for i in range(len(mutation_order) - 1):
            curr_mut = mutation_order[i]
            next_mut = mutation_order[i + 1]
            
            # Create unique node identifiers using position and amino acid
            curr_node = (pos1[curr_mut], possible_mutations[curr_mut][0])
            next_node = (pos1[next_mut], possible_mutations[next_mut][0])
            edge = (curr_node, next_node)
            
            if edge not in edge_counts:
                edge_counts[edge] = 0
                edge_aucs[edge] = 0
            edge_counts[edge] += 1
            edge_aucs[edge] += pathway_auc
    
    # Prepare data for Cytoscape export
    nodes_data = []
    edges_data = []
    
    # Add edges that appear frequently enough
    for (node1, node2), count in edge_counts.items():
        frequency = count / total_pathways
        if frequency >= min_frequency:
            # Create node labels
            node1_label = f"{node1[0]}_{node1[1]}"
            node2_label = f"{node2[0]}_{node2[1]}"
            
            # Add nodes data
            if node1_label not in [n['id'] for n in nodes_data]:
                nodes_data.append({
                    'id': node1_label,
                    'position': node1[0],
                    'amino_acid': node1[1]
                })
            if node2_label not in [n['id'] for n in nodes_data]:
                nodes_data.append({
                    'id': node2_label,
                    'position': node2[0],
                    'amino_acid': node2[1]
                })
            
            # Add edge data
            edges_data.append({
                'source': node1_label,
                'target': node2_label,
                'weight': edge_aucs[(node1, node2)],
                'frequency': frequency
            })
            
            # Add edge to NetworkX graph for visualization
            G.add_edge(node1_label, node2_label, 
                      weight=edge_aucs[(node1, node2)],
                      frequency=frequency)
    
    # Create DataFrames for Cytoscape
    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)
    
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # Draw edges with width proportional to total AUC
    edges = G.edges()
    weights = [G[u][v]['weight'] / max(edge_aucs.values()) * 5 for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                          arrowsize=20)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Consensus Mutation Order Graph\n(Edge width indicates pathway AUC)")
    plt.axis('off')
    
    return plt, nodes_df, edges_df

def train_markov_episodes(env, episodes, tracker, possible_mutations, batch_size=16):
    episode_times = []
    device_target_mean_embedding = env.target_mean_embedding.to(device)

    for episode in range(episodes):
        episode_start = time()
        print(f"\nStart episode {episode}")
        
        current_mutstate = env.reset()
        #available_actions = list(range(len(possible_mutations)))
        available_actions = [i for i in range(len(possible_mutations)) 
                           if possible_mutations[i][0] != possible_mutations[i][1]]

        random.shuffle(available_actions)
        
        episode_similarities = torch.tensor([], device=device)
        episode_actions = []
        
        # Pre-compute all sequences and tokens for this episode
        all_sequences = []
        for action in available_actions:
            current_mutstate[action] = 1
            sequence, _ = create_substitution_mask(current_mutstate, possible_mutations)
            all_sequences.append(" ".join(sequence))
            
        # Single tokenization and device transfer for all sequences
        all_tokens = tokenizer(
            all_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Process batches using pre-computed tokens
        for i in range(0, len(available_actions), batch_size):
            batch_end = min(i + batch_size, len(available_actions))
            batch_actions = available_actions[i:batch_end]
            
            # Get pre-computed tokens for this batch
            batch_tokens = {
                "input_ids": all_tokens["input_ids"][i:batch_end]
            }
            
            # Get embeddings
            with torch.no_grad():
                encoder_output = model.encoder(
                    input_ids=batch_tokens["input_ids"],
                    output_attentions=False
                )
                batch_embeddings = torch.mean(encoder_output.hidden_states[-1], dim=1)
                batch_similarities = F.cosine_similarity(
                    batch_embeddings, 
                    device_target_mean_embedding
                )
            
            # Store results
            episode_similarities = torch.cat([episode_similarities, batch_similarities])
            episode_actions.extend(batch_actions)
        
        # Clear cache after full episode
        torch.cuda.empty_cache()
        gc.collect()
        
        # Only move to CPU once at the end of episode
        episode_similarities_np = episode_similarities.cpu().numpy()
        for step, (action, similarity) in enumerate(zip(episode_actions, episode_similarities_np)):
            tracker.update(action, episode, step, similarity)
        
        episode_time = time() - episode_start
        episode_times.append(episode_time)
        print(f"Episode {episode} completed in {episode_time:.3f}s")

    return tracker.get_step_log()

def plot_episode_trajectories(step_log, output_dir):
    """
    Plot all episode trajectories showing how cosine similarity changes with mutations.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    
    # Ensure similarity_score is numeric
    step_log['similarity_score'] = pd.to_numeric(step_log['similarity_score'], errors='coerce')
    
    # Get unique episodes
    episodes = step_log['episode'].unique()
    
    # Plot each episode's trajectory in light gray
    for episode in episodes:
        episode_data = step_log[step_log['episode'] == episode]
        plt.plot(range(len(episode_data)), 
                episode_data['similarity_score'].values,  # Use .values to ensure we have a 1D array
                color='lightgray', 
                alpha=0.5,
                linewidth=2)
    
    plt.xlabel('Mutation Order')
    plt.ylabel('Cosine Similarity to Target')
    plt.title('Mutation Trajectories Across Episodes')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the plot
    plt.savefig(f'{output_dir}/mutation_trajectories.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def print_model_info(model):
    """Print detailed model information"""
    print("\nModel Information:")
    print(f"Model type: {type(model)}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"Using half precision: {next(model.parameters()).dtype == torch.float16}")
    print(f"Device: {next(model.parameters()).device}")
    
    if hasattr(model, 'encoder'):
        print("\nEncoder Information:")
        print(f"Number of layers: {len(model.encoder.block)}")
        print(f"Hidden size: {model.encoder.block[0].layer[0].SelfAttention.q.weight.shape[0]}")

def plot_pathway_progressions(pathway_df, output_dir, original_seqsims):
    plt.figure(figsize=(12, 6))
    
# Process each pathway
    for idx, row in pathway_df.iterrows():
        seqsim_prog = row['seqsim_progression']
        initial_seqsim = original_seqsims[0] 
        # Prepend the initial sequence similarity
        full_progression = [initial_seqsim] + seqsim_prog
        steps = range(len(full_progression))
            
        # Plot line with transparency
        plt.plot(steps, full_progression, alpha=0.5, color='blue')
        
    # Customize plot
    plt.xlabel('Mutation Steps')
    plt.ylabel('Sequence Similarity')
    plt.title('Sequence Similarity Progression by Pathway')
    plt.grid(True, alpha=0.3)
        
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pathway_progressions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pathway_progressions.pdf')
    plt.close()



    """Plot the progression of sequence similarities for each pathway."""
    # Use original_seqsims directly instead of recalculating
    steps = range(len(original_seqsims))
    plt.plot(steps,
            original_seqsims,
            color='black',
            linewidth=2,
            linestyle='--',
            label='Best Path',
            zorder=2)
    
    # ... rest of plotting code remains the same ...

def optimize_mutation_path(best_path, possible_mutations, model, tokenizer, device, target_ids, window_size=2):
    """
    Optimize a mutation path by trying local permutations within a sliding window.
    
    Args:
        best_path (list): List of mutation indices representing the current best path
        window_size (int): Size of the window to consider for local permutations
    """
    from itertools import permutations
    
    best_seqsims, _ = apply_actions_in_order(best_path, possible_mutations, target_ids, model, tokenizer, device)
    best_score = max(best_seqsims)
    improved = True
    
    while improved:
        improved = False
        
        # Slide window over the path
        for i in range(len(best_path) - window_size + 1):
            window = best_path[i:i + window_size]
            
            # Try all permutations within the window
            for perm in permutations(window):
                new_path = best_path[:i] + list(perm) + best_path[i + window_size:]
                seqsims, _ = apply_actions_in_order(new_path, possible_mutations, target_ids, model, tokenizer, device)
                
                if max(seqsims) > best_score:
                    best_path = new_path
                    best_score = max(seqsims)
                    improved = True
                    break
                    
            if improved:
                break
                
    return best_path, best_score

def beam_search_optimization(best_path, possible_mutations, model, tokenizer, device, target_ids, beam_width=3):
    """
    Use beam search to optimize the mutation path by maintaining multiple candidate paths.
    """
    candidates = [(best_path, None)]  # (path, score) tuples
    
    for i in range(len(best_path)):
        new_candidates = []
        
        for path, _ in candidates:
            # Try inserting each remaining mutation at position i
            remaining_mutations = set(range(len(possible_mutations))) - set(path[:i])
            
            for mutation in remaining_mutations:
                new_path = path[:i] + [mutation] + path[i+1:]
                seqsims, _ = apply_actions_in_order(new_path[:i+1], possible_mutations, target_ids, model, tokenizer, device)
                new_candidates.append((new_path, max(seqsims)))
        
        # Keep top beam_width candidates
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return candidates[0][0], candidates[0][1]

def combine_good_paths(pathway_df, possible_mutations, model, tokenizer, device, target_ids, n_best=3):
    """
    Attempt to combine the best parts of multiple successful paths.
    """
    # Get top n paths
    top_paths = pathway_df.nlargest(n_best, 'final_auc')
    
    # Find common prefixes among good paths
    common_prefixes = []
    for length in range(1, max(len(path) for path in top_paths['mutation_order'])):
        prefixes = [tuple(path[:length]) for path in top_paths['mutation_order']]
        if len(set(prefixes)) == 1:
            common_prefixes.append(list(prefixes[0]))
    
    # Try extending common prefixes with different suffixes
    best_path = None
    best_score = float('-inf')
    
    for prefix in common_prefixes:
        for path in top_paths['mutation_order']:
            # Create new path combining prefix with unique suffix
            suffix = [m for m in path if m not in prefix]
            combined_path = prefix + suffix
            
            seqsims, _ = apply_actions_in_order(combined_path, possible_mutations, target_ids, model, tokenizer, device)
            score = max(seqsims)
            
            if score > best_score:
                best_score = score
                best_path = combined_path
                
    return best_path, best_score

def format_mutation(mutation_tuple, pos1, pos2):
    """Format a mutation tuple into a readable string.
    
    Args:
        mutation_tuple: Tuple of (original_AA, target_AA)
        pos1: List of positions in original sequence
        pos2: List of positions in target sequence
    
    Returns:
        str: Formatted string like 'AAorig{pos1}_AAtarget{pos2}'
    """
    orig_aa, target_aa = mutation_tuple
    idx = possible_mutations.index(mutation_tuple)
    return f"{orig_aa}{pos1[idx]}_{target_aa}{pos2[idx]}"

def optimize_best_pathways(pathway_df, target_ids, model, tokenizer, device, n_best=5, window_size=2, beam_width=3):
    """Optimize the best pathways by trying different mutation orders using beam search within windows.
    
    Args:
        pathway_df: DataFrame containing mutation pathways
        target_ids: Target sequence IDs
        model: The protein language model
        tokenizer: The tokenizer
        device: Device to run computations on
        n_best: Number of best pathways to return (default: 5)
        window_size: Size of window for local optimization (default: 2)
        beam_width: Number of best partial paths to keep at each step (default: 3)
    """
    print("\nOptimizing mutation pathways...")
    
    # Print initial pathway information
    print(f"\nInitial pathway analysis summary:")
    print(f"Number of pathways: {len(pathway_df)}")
    print(f"AUC range: {pathway_df['final_auc'].min():.4f} to {pathway_df['final_auc'].max():.4f}")
    
    # Sort by final_auc and get the best pathway
    sorted_df = pathway_df.sort_values('final_auc', ascending=False)
    best_path = sorted_df.iloc[0]['mutation_order']
    
    print(f"\nBest original pathway:")
    print(f"AUC: {sorted_df.iloc[0]['final_auc']:.4f}")
    print(f"Mutation order: {best_path}")
    print(f"Final similarity: {sorted_df.iloc[0]['seqsim_progression'][-1]:.4f}")
    print(f"Using window size: {window_size}")
    
    # Try different permutations of the best path using sliding windows
    optimization_results = []
    path_length = len(best_path)
    
    for i in range(n_best):
        current_path = best_path.copy()
        
        # Optimize within sliding windows
        for start_idx in range(0, path_length - window_size + 1):
            window = current_path[start_idx:start_idx + window_size]
            
            # Keep track of best beam_width sequences for this window
            beam_candidates = [(window, -float('inf'))]  # (sequence, score) pairs
            
            # Try different permutations and keep top beam_width
            for _ in range(3):  # Number of random permutations to try
                shuffled_window = window.copy()
                random.shuffle(shuffled_window)
                
                # Create temporary path with shuffled window
                temp_path = current_path.copy()
                temp_path[start_idx:start_idx + window_size] = shuffled_window
                
                # Apply mutations and get results
                seqsims, seqs = apply_actions_in_order(
                    temp_path, possible_mutations, target_ids, model, tokenizer, device
                )
                
                # Calculate AUC for this path
                auc = np.trapz(seqsims)
                
                # Add to beam candidates
                beam_candidates.append((shuffled_window, auc))
                
                # Keep only top beam_width candidates
                beam_candidates.sort(key=lambda x: x[1], reverse=True)
                beam_candidates = beam_candidates[:beam_width]
            
            # Update current path with best window arrangement
            current_path[start_idx:start_idx + window_size] = beam_candidates[0][0]
        
        # Get final results for the optimized path
        final_seqsims, final_seqs = apply_actions_in_order(
            current_path, possible_mutations, target_ids, model, tokenizer, device
        )
        
        final_auc = np.trapz(final_seqsims)
        mutation_strings = [format_mutation(possible_mutations[idx], pos1, pos2) 
                          for idx in current_path]
        
        optimization_results.append({
            'mutation_order': current_path,
            'mutation_strings': mutation_strings,
            'seqsim_progression': final_seqsims,
            'sequences': final_seqs,
            'auc': final_auc
        })
        
        print(f"\nOptimization attempt {i+1}:")
        print(f"AUC: {final_auc:.4f}")
        print(f"Final similarity: {final_seqsims[-1]:.4f}")
        print("Mutations:")
        for mut in mutation_strings:
            print(f"  {mut}")
    
    # Sort optimization results by AUC
    optimization_results.sort(key=lambda x: x['auc'], reverse=True)
    
    print("\nOptimization summary:")
    print(f"Best optimized AUC: {optimization_results[0]['auc']:.4f}")
    print(f"Original best AUC: {sorted_df.iloc[0]['final_auc']:.4f}")
    print(f"Improvement: {(optimization_results[0]['auc'] - sorted_df.iloc[0]['final_auc']):.4f}")
    
    return best_path, optimization_results

def plot_pathway_comparison(pathway_df, optimization_results, output_dir, starting_seqsim):
    """Create plot comparing original pathways with optimized paths.
    
    Args:
        pathway_df: DataFrame containing original pathways
        optimization_results: List of optimized pathway results
        output_dir: Directory to save plots
        starting_seqsim: Initial sequence similarity before any mutations
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original pathways in light gray
    for _, pathway in pathway_df.iterrows():
        # Prepend starting similarity
        full_progression = [starting_seqsim] + list(pathway['seqsim_progression'])
        steps = range(len(full_progression))
        
        plt.plot(steps, 
                full_progression,
                color='lightgray', 
                alpha=0.3,
                linewidth=1,
                zorder=1)
    
    # Plot optimized paths with distinct colors
    cmap = plt.cm.viridis(np.linspace(0, 1, len(optimization_results)))
    for idx, result in enumerate(optimization_results):
        # Prepend starting similarity
        full_progression = [starting_seqsim] + list(result['seqsim_progression'])
        steps = range(len(full_progression))
        
        plt.plot(steps,
                full_progression,
                color=cmap[idx],
                linewidth=2,
                label=f'Optimized Path {idx+1}',
                zorder=2)
    
    plt.xlabel('Mutation Step')
    plt.ylabel('Sequence Similarity')
    plt.title('Comparison of Original vs Optimized Mutation Pathways')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{output_dir}/pathway_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.savefig(f'{output_dir}/pathway_comparison.pdf',
                bbox_inches='tight')
    plt.close()
    
    # Create second plot with best original path
    plt.figure(figsize=(12, 8))
    
    # Get best original pathway
    best_original = pathway_df.nlargest(1, 'final_auc').iloc[0]
    
    # Plot best original pathway with starting point
    full_progression = [starting_seqsim] + list(best_original['seqsim_progression'])
    steps = range(len(full_progression))
    plt.plot(steps,
            full_progression,
            color='black',  # Changed to black
            linewidth=3,    # Made thicker
            linestyle='--', # Made dashed
            label='Best Original Path',
            zorder=1)
    
    # Plot optimized paths with markers
    markers = ['o', 's', 'D', '^', 'v']  # Different marker shapes
    for idx, result in enumerate(optimization_results):
        full_progression = [starting_seqsim] + list(result['seqsim_progression'])
        steps = range(len(full_progression))
        plt.plot(steps,
                full_progression,
                color=cmap[idx],
                linewidth=2,
                marker=markers[idx % len(markers)],  # Add markers
                markersize=6,
                alpha=0.5,    # Slight transparency
                label=f'Optimized Path {idx+1}',
                zorder=2)
    
    plt.xlabel('Mutation Step')
    plt.ylabel('Sequence Similarity')
    plt.title('Best Original Path vs Optimized Paths')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/best_pathway_comparison.png',
                dpi=300,
                bbox_inches='tight')
    plt.savefig(f'{output_dir}/best_pathway_comparison.pdf',
                bbox_inches='tight')
    plt.close()

def beam_search_refinement(original_path, possible_mutations, target_ids, model, tokenizer, device, 
                         beam_width=3, look_ahead=5, starting_seqsim=None, original_aucs=None,
                         max_steps=None):  # Add max_steps parameter
    """
    Beam search to explore mutation paths
    """
    current_beam = [([], 0.0, [starting_seqsim])]  # (path, score, seqsims)
    all_level_bests = []
    
    # Determine number of levels to process
    n_levels = min(len(original_path), max_steps) if max_steps else len(original_path)
    
    for level in range(n_levels):  # Modified to use n_levels instead of full path length
        new_candidates = []
        print(f"\nProcessing level {level + 1}")
        print(f"Original AUC at this level: {original_aucs[level+1]:.4f}")
        
        for current_path, current_score, current_seqsims in current_beam:
            remaining = [m for m in original_path if m not in current_path]
            next_mutations = remaining[:look_ahead]
            
            for next_mut in next_mutations:
                new_path = current_path + [next_mut]
                
                seqsims, _ = apply_actions_in_order(
                    new_path,
                    possible_mutations,
                    target_ids,
                    model,
                    tokenizer,
                    device,
                    only_final=True
                )
                
                new_seqsims = current_seqsims + [seqsims[-1]]
                auc = np.trapz(new_seqsims)
                new_candidates.append((new_path, auc, new_seqsims))
        
        # Keep best paths by AUC
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        current_beam = new_candidates[:beam_width]
        all_level_bests.append(current_beam)
        
        # Print progress focusing on AUC comparison
        print(f"\nLevel {level + 1} beam search results:")
        for i, (path, auc, _) in enumerate(current_beam):
            print(f"  Path {i+1}: AUC = {auc:.4f}")
            print(f"    Difference from original AUC: {auc - original_aucs[level+1]:+.4f}")
    
    return all_level_bests

def analyze_level_results(all_level_bests, possible_mutations):
    """
    Analyzes the results from each level of the beam search.
    
    Args:
        all_level_bests: List of best paths at each level
        possible_mutations: List of possible mutations
    
    Returns:
        pd.DataFrame: Analysis of paths at each level
    """
    results = []
    
    for level, level_paths in enumerate(all_level_bests):
        for rank, (path, score, seqsims) in enumerate(level_paths):
            results.append({
                'level': level + 1,
                'rank': rank + 1,
                'score': score,
                'path_length': len(path),
                'mutations': [format_mutation(possible_mutations[m], pos1, pos2) for m in path],
                'path': path,
                'seqsims': seqsims
            })
    
    return pd.DataFrame(results)

def refine_mutation_path(pathway_df, possible_mutations, target_ids, model, tokenizer, device,
                        beam_width=3, look_ahead=5, max_steps=None, original_seqsims=None, original_aucs=None):
    """
    Refines the best mutation paths using beam search.
    If max_steps is specified, optimizes only the first n steps and keeps the rest of the original path.
    """
    # Get the best path from the original analysis
    best_original = pathway_df.nlargest(1, 'final_auc').iloc[0]
    original_path = best_original['mutation_order']
    
    print("\nStarting beam search refinement...")
    print(f"Original path length: {len(original_path)}")
    if max_steps:
        print(f"Optimizing first {max_steps} steps only")
        print(f"Remaining {len(original_path) - max_steps} steps will keep original path")
    print("Original mutations:", [format_mutation(possible_mutations[m], pos1, pos2) 
                                for m in original_path[:5]], "...")
    print(f"Using beam width: {beam_width}, look ahead: {look_ahead}")
    
    # Perform beam search refinement with passed parameters
    all_level_bests = beam_search_refinement(
        original_path=original_path,
        possible_mutations=possible_mutations,
        target_ids=target_ids,
        model=model,
        tokenizer=tokenizer,
        device=device,
        beam_width=beam_width,
        look_ahead=look_ahead,
        starting_seqsim=original_seqsims[0],  # Use first value from progression
        original_aucs=original_aucs,
        max_steps = max_steps
    )
    
    # Get the best path from the final optimized level
    final_level = all_level_bests[-1]
    optimized_path = final_level[0][0]  # Get path from first tuple in final level
    
    # If using max_steps, combine optimized initial path with remaining original path
    if max_steps:
        remaining_mutations = [m for m in original_path if m not in optimized_path]
        best_final_path = optimized_path + remaining_mutations
        print("\nCombined path:")
        print(f"First {max_steps} steps: Optimized")
        print(f"Remaining {len(remaining_mutations)} steps: Original path")
    else:
        best_final_path = optimized_path
    
    # Analyze results at each level
    level_results_df = analyze_level_results(all_level_bests, possible_mutations)
    
    return level_results_df, all_level_bests, best_final_path

def plot_beam_search_comparison(pathway_df, level_results_df, output_dir, starting_seqsim, original_seqsims):
    """Create plots comparing original pathways with beam search optimized paths."""
    plt.figure(figsize=(12, 8))
    
    # Plot original pathways in light gray
    for _, pathway in pathway_df.iterrows():
        # Use original_seqsims directly for best original path
        steps = range(len(original_seqsims))
        plt.plot(steps,
                original_seqsims,
                color='black',
                linewidth=3,
                linestyle='--',
                label='Best Original Path',
                zorder=1)
    
    # Plot beam search paths with distinct colors
    # Get unique levels and create color map
    levels = level_results_df['level'].unique()
    cmap = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    
    for idx, level in enumerate(levels):
        level_data = level_results_df[level_results_df['level'] == level]
        best_path = level_data.iloc[0]  # Get the best path at this level
        
        # Prepend starting similarity
        full_progression = [starting_seqsim] + list(best_path['seqsims'])
        steps = range(len(full_progression))
        
        plt.plot(steps,
                full_progression,
                color=cmap[idx],
                linewidth=2,
                label=f'Level {level}',
                zorder=2)
    
    plt.xlabel('Mutation Step')
    plt.ylabel('Sequence Similarity')
    plt.title('Comparison of Original vs Beam Search Optimized Paths')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{output_dir}/beam_search_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.savefig(f'{output_dir}/beam_search_comparison.pdf',
                bbox_inches='tight')
    plt.close()
    
    # Create second plot focusing on best paths
    plt.figure(figsize=(12, 8))
    
    # Get best original pathway
    best_original = pathway_df.nlargest(1, 'final_auc').iloc[0]
    
    # Plot best original pathway with starting point
    full_progression = [starting_seqsim] + list(best_original['seqsim_progression'])
    steps = range(len(full_progression))
    plt.plot(steps,
            full_progression,
            color='black',
            linewidth=3,
            linestyle='--',
            label='Best Original Path',
            zorder=1)
    
    # Plot beam search paths - don't prepend starting_seqsim since it's already included
    levels = level_results_df['level'].unique()
    cmap = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    markers = ['o', 's', 'D', '^', 'v']  # Different marker shapes
    
    for idx, level in enumerate(levels):
        level_data = level_results_df[level_results_df['level'] == level]
        best_path = level_data.iloc[0]
        
        # Use seqsims directly - it already includes starting_seqsim
        steps = range(len(best_path['seqsims']))
        plt.plot(steps,
                best_path['seqsims'],  # Don't prepend starting_seqsim
                color=cmap[idx],
                linewidth=2,
                marker=markers[idx % len(markers)],
                markersize=6,
                alpha=0.7,
                label=f'Level {level}',
                zorder=2)
    
    plt.xlabel('Mutation Step')
    plt.ylabel('Sequence Similarity')
    plt.title('Comparison of Original vs Beam Search Optimized Paths')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{output_dir}/beam_search_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.savefig(f'{output_dir}/beam_search_comparison.pdf',
                bbox_inches='tight')
    plt.close()

def plot_beam_search_levels(level_results_df, output_basename):
    """Create plot showing sequence similarities for best paths at each level."""
    plt.figure(figsize=(12, 8))
    
    # Get unique levels and create color map
    levels = level_results_df['level'].unique()
    cmap = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    
    # Plot each level's best paths
    for idx, level in enumerate(levels):
        level_data = level_results_df[level_results_df['level'] == level]
        
        # Plot each path at this level
        for rank, (_, path_data) in enumerate(level_data.iterrows()):
            # Use seqsims directly - it already includes starting_seqsim
            steps = range(len(path_data['seqsims']))
            
            # Different line styles for different ranks within same level
            linestyle = '-' if rank == 0 else '--' if rank == 1 else ':'
            
            # Fix: Ensure alpha stays between 0 and 1
            alpha = max(0.2, 0.7 - (rank * 0.15))  # Start at 0.7, decrease by 0.15, minimum 0.2
            
            plt.plot(steps,
                    path_data['seqsims'],  # Don't prepend starting_seqsim
                    color=cmap[idx],
                    linewidth=2,
                    linestyle=linestyle,
                    alpha=alpha)
    
    plt.xlabel('Mutation Step')
    plt.ylabel('Sequence Similarity')
    plt.title('Sequence Similarities of Best Paths at Each Level')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save plots using the output_basename
    plt.savefig(f'{output_basename}_level_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.savefig(f'{output_basename}_level_comparison.pdf',
                bbox_inches='tight')
    plt.close()

def get_original_progressions(pathway_df, starting_seqsim):
    """
    Calculate original seqsim and AUC progressions for the best path.
    
    Args:
        pathway_df: DataFrame containing pathway analysis results
        starting_seqsim: Initial sequence similarity
        
    Returns:
        tuple: (original_seqsims, original_aucs) where both include starting point
    """
    # Get best original path
    best_original = pathway_df.nlargest(1, 'final_auc').iloc[0]
    
    # Create full seqsim progression including starting point
    original_seqsims = [starting_seqsim] + list(best_original['seqsim_progression'])
    
    # Calculate AUCs at each level
    original_aucs = [np.trapz(original_seqsims[:i+1]) for i in range(len(original_seqsims))]
    
    return original_seqsims, original_aucs

def iterative_beam_search(pathway_df, possible_mutations, target_ids, model, tokenizer, device,
                         beam_width=3, look_ahead=5, max_steps=None, n_iterations=3,
                         original_seqsims=None, original_aucs=None):
    """
    Performs multiple iterations of beam search optimization, using the best path
    from each iteration as the starting point for the next.
    """
    print(f"\nStarting iterative beam search with {n_iterations} iterations...")
    
    # Store results from all iterations
    iteration_results = []
    best_path = None
    best_score = float('-inf')
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # For first iteration, use original pathway
        if iteration == 0:
            current_pathway_df = pathway_df
        else:
            # Create new pathway_df using best path from previous iteration
            current_pathway_df = pd.DataFrame([{
                'mutation_order': best_path,
                'final_auc': best_score,
                'seqsim_progression': seqsims
            }])
        
        # Run beam search refinement
        level_results_df, all_level_bests, current_best_path = refine_mutation_path(
            pathway_df=current_pathway_df,
            possible_mutations=possible_mutations,
            target_ids=target_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            beam_width=beam_width,
            look_ahead=look_ahead,
            max_steps=max_steps,
            original_seqsims=original_seqsims,
            original_aucs=original_aucs
        )
        
        # Create level plot for this iteration
        plot_beam_search_levels(level_results_df, 
                              f"{output_dir}/iteration_{iteration + 1}")
        
        # Apply current best path to get scores
        seqsims, _ = apply_actions_in_order(
            current_best_path,
            possible_mutations,
            target_ids,
            model,
            tokenizer,
            device
        )
        current_score = np.trapz(seqsims)
        
        # Store iteration results
        iteration_results.append({
            'iteration': iteration + 1,
            'level_results': level_results_df,
            'best_path': current_best_path,
            'score': current_score,
            'seqsims': seqsims
        })
        
        # Update best overall if current is better
        if current_score > best_score:
            best_score = current_score
            best_path = current_best_path.copy()
            print(f"New best score: {best_score:.4f}")
        else:
            print(f"No improvement: {current_score:.4f} vs best {best_score:.4f}")
            
        # Plot progress after each iteration
        plot_iterative_beam_search_progress(iteration_results, output_dir, original_seqsims)
        
    return iteration_results, best_path, best_score

def plot_iterative_beam_search_progress(iteration_results, output_dir, original_seqsims):
    """Plot the progression of sequence similarities across iterations."""
    plt.figure(figsize=(12, 8))
    
    # Plot original sequence
    steps = range(len(original_seqsims))
    plt.plot(steps,
            original_seqsims,
            color='black',
            linewidth=2,
            linestyle='--',
            label='Original',
            zorder=1)
    
    # Plot each iteration with different colors
    cmap = plt.cm.viridis(np.linspace(0, 1, len(iteration_results)))
    for idx, result in enumerate(iteration_results):
        full_progression = [original_seqsims[0]] + list(result['seqsims'])
        steps = range(len(full_progression))
        

        plt.plot(steps,
                full_progression,
                color=cmap[idx],
                linewidth=2,
                label=f'Iteration {result["iteration"]}',
                zorder=2+idx)
    
    plt.xlabel('Mutation Step')
    plt.ylabel('Sequence Similarity')
    plt.title('Beam Search Optimization Across Iterations')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plots
    plt.savefig(f'{output_dir}/iterative_beam_search_progress.png',
                dpi=300,
                bbox_inches='tight')
    plt.savefig(f'{output_dir}/iterative_beam_search_progress.pdf',
                bbox_inches='tight')
    plt.close()

def plot_mutation_phases(step_log, pathway_df, output_dir, original_seqsims, iteration_results=None, reference_seqs=None):
    """
    Create a figure showing exploration and optimization phases, with optional refinement phase.
    """
    # Determine number of subplots based on whether iteration_results is provided
    n_plots = 4  # Always include reference panel
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    
    # Plot reference sequences if provided
    print(reference_seqs)
    if reference_seqs is not None:  # Changed from if reference_seqs:
        for name, sim_score in reference_seqs.items():
            # Add horizontal line across first three subplots
            for ax in axes[:-1]:  # Exclude last panel
                ax.axhline(y=sim_score, color='green', linestyle=':', alpha=0.5,
                          label=f'Reference: {name}')
    
    # 1. Exploration Phase (Mutation Trajectories)
    episodes = step_log['episode'].unique()
    for episode in episodes:
        episode_data = step_log[step_log['episode'] == episode]
        axes[0].plot(range(len(episode_data)), 
                    episode_data['similarity_score'].values,
                    color='lightgray', 
                    alpha=0.5,
                    linewidth=1)
    axes[0].set_title('Phase 1: Exploration\n(All Mutation Trajectories)')
    axes[0].set_xlabel('Mutation Order')
    axes[0].set_ylabel('Sequence Similarity')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # 2. Optimization Phase (Pathway Progressions)
    for _, pathway in pathway_df.iterrows():
        full_progression = [original_seqsims[0]] + list(pathway['seqsim_progression'])
        steps = range(len(full_progression))
        axes[1].plot(steps, 
                    full_progression,
                    color='lightgray', 
                    alpha=0.3,
                    linewidth=1)
    
    best_original = pathway_df.nlargest(1, 'final_auc').iloc[0]
    full_progression = [original_seqsims[0]] + list(best_original['seqsim_progression'])
    steps = range(len(full_progression))
    axes[1].plot(steps,
                full_progression,
                color='black',
                linewidth=2,
                linestyle='--',
                label='Best Path')
    axes[1].set_title('Phase 2: Optimization\n(Selected Pathways)')
    axes[1].set_xlabel('Mutation Order')
    axes[1].set_ylabel('Sequence Similarity')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # 3. Refinement Phase (if iteration_results provided)
    if iteration_results is not None:
        steps = range(len(original_seqsims))
        axes[2].plot(steps,
                    original_seqsims,
                    color='black',
                    linewidth=2,
                    linestyle='--',
                    label='Original Best',
                    zorder=1)
        
        cmap = plt.cm.viridis(np.linspace(0, 1, len(iteration_results)))
        for idx, result in enumerate(iteration_results):
            full_progression = [original_seqsims[0]] + list(result['seqsims'])
            steps = range(len(full_progression))
            axes[2].plot(steps,
                        full_progression,
                        color=cmap[idx],
                        linewidth=2,
                        label=f'Iteration {result["iteration"]}',
                        zorder=2+idx)
        
        axes[2].set_title('Phase 3: Refinement\n(Beam Search Iterations)')
        axes[2].set_xlabel('Mutation Order')
        axes[2].set_ylabel('Sequence Similarity')
        axes[2].grid(True, linestyle='--', alpha=0.3)
        axes[2].legend()
    else:
        axes[2].set_visible(False)  # Hide if no iteration results
    
    # 4. Reference Sequences Panel
    if reference_seqs is not None:  # Changed from if reference_seqs:
        # Sort reference sequences by similarity score
        sorted_refs = dict(sorted(reference_seqs.items(), key=lambda x: x[1], reverse=True))
        
        # Create bar plot of reference sequence similarities
        names = list(sorted_refs.keys())
        scores = list(sorted_refs.values())
        
        # Create bar plot
        bars = axes[3].bar(range(len(names)), scores, color='lightgreen')
        
        # Customize the plot
        axes[3].set_title('Reference Sequences\nSimilarity Scores')
        axes[3].set_xlabel('Reference Sequences')
        axes[3].set_ylabel('Sequence Similarity')
        
        # Rotate x-axis labels for better readability
        axes[3].set_xticks(range(len(names)))
        axes[3].set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
        
        # Add grid for better readability
        axes[3].grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Set y-axis limits to match other panels
        axes[3].set_ylim(0, 1.0)
    else:
        axes[3].text(0.5, 0.5, 'No reference\nsequences provided',
                    ha='center', va='center',
                    transform=axes[3].transAxes)
        axes[3].set_title('Reference Sequences')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mutation_phases_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/mutation_phases_analysis.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    
    if not args.fasta_file and not args.aln_file:
        print("Provide fasta or alignment")
        exit(1)
    
    # Setup model and tokenizer
    clear_gpu_memory()
    model, tokenizer, model_type, device = setup_model_and_tokenizer(args.model_path)
    outfile_base = "testmut" 
    
    # Setup sequences and environment
    (orig_seqrecord, target_seqrecord, orig_name, target_name, 
     orig_seq_nospaces, target_seq_nospaces, alignment, 
     output_dir, filename) = setup_sequences(args, outfile_base)
    
    # Tokenize sequences
    tokens_orig = tokenizer(" ".join(orig_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
    orig_ids = tokens_orig["input_ids"].to(device)
    tokens_target = tokenizer(" ".join(target_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
    target_ids = tokens_target["input_ids"].to(device)
    
    # Calculate initial embeddings and similarities
    with torch.no_grad():
        target_mean_embedding = get_representation(model, target_ids, model_type, layers=[-1], output_attentions=False)
        orig_mean_embedding = get_representation(model, orig_ids, model_type, layers=[-1], output_attentions=False)
        
    # Calculate starting similarity (needed for all paths)
    starting_seqsim = float(F.cosine_similarity(orig_mean_embedding, target_mean_embedding).cpu().numpy())

    # Process reference sequences if provided
    reference_seqs = None
    if args.reference_fasta:
        reference_seqs = {}
        with open(args.reference_fasta, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Tokenize sequence
                tokens = tokenizer(" ".join(str(record.seq)), 
                                 return_tensors="pt", 
                                 padding=True, 
                                 truncation=True).to(device)
                
                # Get similarity score
                with torch.no_grad():
                    ref_embedding = get_representation(model, 
                                                     tokens["input_ids"], 
                                                     model_type, 
                                                     layers=[-1], 
                                                     output_attentions=False)
                    sim_score = F.cosine_similarity(ref_embedding, 
                                                  target_mean_embedding).item()
                reference_seqs[record.id] = sim_score

    # Setup mutations
    possible_mutations, pos1, pos2 = generate_differences_and_positions(alignment[-1], alignment[0])
    print(possible_mutations, pos1, pos2)

    # PATHWAY 1: Run everything from scratch
    if not args.step_log and not args.pathway_file:
        print("\nRunning new trials...")
        env = ProteinMutationEnv(orig_ids.squeeze(), 
                                target_ids.squeeze(), 
                                possible_mutations, 
                                model, 
                                tokenizer, 
                                device,
                                baseline_reward=args.baseline_reward,  # Add this parameter
                                starting_seqsim=starting_seqsim)  # Add this parameter
        tracker = ActionValueTracker(starting_seqsim=starting_seqsim)
        step_log = train_markov_episodes(env, args.episodes, tracker, possible_mutations)
        
        step_log_filename = os.path.join(output_dir, f'step_log_{args.episodes}ep.csv')
        step_log.to_csv(step_log_filename, index=False)
        print(f"Saved step_log to {step_log_filename}")

        plot_episode_trajectories(step_log, output_dir)
        
        # Proceed to pathway analysis
        print("\nAnalyzing mutation pathways...")
        pathway_df = analyze_mutation_pathways(
            total_rewards=step_log,
            possible_mutations=possible_mutations,
            target_ids=target_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_subsamples=args.n_subsamples,
            subsample_size=args.subsample_size
        )
        pathway_df.to_csv(f'{output_dir}/mutation_pathway_analysis.csv', index=False)

    # PATHWAY 2: Start from step_log
    elif args.step_log and not args.pathway_file:
        print(f"\nLoading previous step log from {args.step_log}")
        step_log = pd.read_csv(args.step_log)
        
        # Proceed to pathway analysis
        print("\nAnalyzing mutation pathways...")
        pathway_df = analyze_mutation_pathways(
            total_rewards=step_log,
            possible_mutations=possible_mutations,
            target_ids=target_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_subsamples=args.n_subsamples,
            subsample_size=args.subsample_size
        )
        
        # Save pathway analysis results
        pathway_df.to_csv(f'{output_dir}/mutation_pathway_analysis.csv', index=False)
        
    # PATHWAY 3: Start from pathway_df
    else:
        print(f"\nLoading previous pathway analysis from {args.pathway_file}")
        pathway_df = pd.read_csv(args.pathway_file)
        
        # Convert string representation of lists back to actual lists
        pathway_df['mutation_order'] = pathway_df['mutation_order'].apply(eval)
        pathway_df['seqsim_progression'] = pathway_df['seqsim_progression'].apply(eval)
        pathway_df['sequences'] = pathway_df['sequences'].apply(eval)

    # Common visualization and analysis for all pathways
    print("\nPlotting pathway progressions...")
    original_seqsims, original_aucs = get_original_progressions(pathway_df, starting_seqsim)
    plot_pathway_progressions(pathway_df, output_dir, original_seqsims)
    
    # Print summary of pathway analysis
    print(f"\nAnalyzed {len(pathway_df)} different mutation pathways")
    print("\nTop 5 pathways by final AUC:")
    top_pathways = pathway_df.nlargest(5, 'final_auc')
    for idx, pathway in top_pathways.iterrows():
        print(f"\nPathway {idx + 1}")
        print(f"Final AUC: {pathway['final_auc']}")
        print("Mutation order:")
        for i, mutation in enumerate(pathway['mutation_order'][:10]):
            print(f"  {i+1}. {possible_mutations[mutation][0]} -> {possible_mutations[mutation][1]}")
        if len(pathway['mutation_order']) > 10:
            print("  ...")

    # Create and save consensus graph
    print("\nCreating consensus mutation graph...")
    plt, nodes_df, edges_df = create_consensus_graph(
        pathway_df=pathway_df,
        pos1=pos1,
        possible_mutations=possible_mutations,
        min_frequency=0.0  # Include edges that appear in at least x% of pathways
    )
    nodes_df.to_csv(f'{output_dir}/consensus_graph_nodes.csv', index=False)
    edges_df.to_csv(f'{output_dir}/consensus_graph_edges.csv', index=False)
    plt.savefig(f'{output_dir}/consensus_mutation_graph.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(args.n_best_paths)
    
   
    # Optimize pathways
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.optimization_method in ['beam', 'both']:
        # Calculate original progressions early
        original_seqsims, original_aucs = get_original_progressions(pathway_df, starting_seqsim)
        
        # Run iterative beam search
        iteration_results, best_path, best_score = iterative_beam_search(
            pathway_df=pathway_df,
            possible_mutations=possible_mutations,
            target_ids=target_ids,
            model=model,
            tokenizer=tokenizer,
            device=device,
            beam_width=args.beam_width,
            look_ahead=args.look_ahead,
            max_steps=args.max_steps,
            n_iterations=args.beam_iterations,
            original_seqsims=original_seqsims,
            original_aucs=original_aucs
        )
        
        # Create three-panel plot including beam search results
        plot_mutation_phases(step_log, pathway_df, output_dir, original_seqsims, iteration_results, reference_seqs=reference_seqs)  # Make sure reference_seqs is being passed
        
        # Save iteration results
        iteration_df = pd.DataFrame([{
            'iteration': r['iteration'],
            'score': r['score'],
            'best_path': r['best_path']
        } for r in iteration_results])
        iteration_df.to_csv(f'{output_dir}/beam_search_iterations_{timestamp}.csv', index=False)


    else:
        plot_mutation_phases(step_log, pathway_df, output_dir, original_seqsims)
 

    print("\nOptimization complete!")
    if args.optimization_method == 'both':
        print("Results from both optimization methods have been saved.")
    elif args.optimization_method == 'original':
        print("Results from window-based optimization have been saved.")
    else:
        print("Results from beam search optimization have been saved.")

    # Apply and save the final optimized path
    final_seqsims, final_seqs = apply_actions_in_order(
        best_path, possible_mutations, target_ids, model, tokenizer, device
    )

    # Create results dataframe with additional mutation information
    results_data = []
    
    # Add initial sequence (original query)
    results_data.append({
        'step': 0,
        'similarity': final_seqsims[0],
        'sequence': orig_seq_nospaces,  # Original query sequence without spaces
        'aa1': '',
        'pos1': '',
        'aa2': '',
        'pos2': '',
        'mutation': ''
    })
    
    # Add each mutation step
    for i, (mutation_idx, seqsim, seq) in enumerate(zip(best_path, final_seqsims[1:], final_seqs[1:]), 1):
        # Get mutation details
        aa1, aa2 = possible_mutations[mutation_idx]
        p1, p2 = pos1[mutation_idx], pos2[mutation_idx]
        
        results_data.append({
            'step': i,
            'similarity': seqsim,
            'aa1': aa1,
            'pos1': p1,
            'aa2': aa2,
            'pos2': p2,
            'mutation': f"{aa1}{p1}_{aa2}{p2}",
            'sequence': seq.replace(" ", "") # Remove spaces from sequence

        })
    
    # Save to CSV
    pd.DataFrame(results_data).to_csv(
        f'{output_dir}/optimized_path_results_{timestamp}.csv', 
        index=False
    )
    
    # Write FASTA file
    with open(f'{output_dir}/optimized_path_sequences_{timestamp}.fasta', 'w') as f:
        for result in results_data:
            # Format header with step, mutation (if any), and similarity
            if result['step'] == 0:
                header = f">step{result['step']}_original_sim{result['similarity']:.3f}"
            else:
                header = f">step{result['step']}_{result['mutation']}_sim{result['similarity']:.3f}"
            
            # Write sequence in FASTA format
            f.write(f"{header}\n{result['sequence']}\n")



# get rank based on current Q-value

#actions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]



