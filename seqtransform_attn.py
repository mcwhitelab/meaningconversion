
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

from transformer_infrastructure.attn_calc import get_attn_data

import pandas as pd
pd.set_option('display.max_rows', 2000)
import random

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from gymnasium import spaces

import numpy as np



# TODO: Fix the imports at some point
import torch.nn as nn
import torch.optim as optim


class ProteinMutationEnv(gym.Env):

    def __init__(self, initial_ids, target_ids, possible_mutations, mutstate, starting_seqsim, starting_mse_self, starting_mse_int, model, tokenizer, device, starting_intface = 0, intface_limits = []):

        super(ProteinMutationEnv, self).__init__()
        self.initial_ids = initial_ids
        self.target_ids = target_ids
        self.possible_mutations = possible_mutations
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.latest_reward = 0  
         
        self.mutstate = mutstate
        #self.best_seqsim, self.tot_attns = self.calculate_reward(self.initial_ids, self.target_ids, self.model, self.tokenizer, self.device)
        self.best_seqsim = starting_seqsim
        self.starting_seqsim = starting_seqsim

        self.lowest_mse_self = starting_mse_self
        self.starting_mse_self = starting_mse_self

        self.lowest_mse_int = starting_mse_int
        self.starting_mse_int = starting_mse_int

        self.starting_intface = starting_intface
        # Define action and observation space

        self.action_space = spaces.Discrete(len(possible_mutations))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(mutstate),), dtype=np.float32)
        #print(self.observation_space)
        self.current_ids = self.initial_ids.clone().detach()
        self.mutation_counter = 0  # Initialize the mutation counter



    def step(self, action):
        # Apply mutation
        # print("STEP --------------------------------------------")
        # this is how we format it:
        # pass the state, then construct the sequence, then tokenize
        # mutstate: [0,1,0]. 
        # Action 1: ['AAAA', ''] # Deletion
        # Action 2: ['A', 'M']   # Substitution
        # Action 3: ['', 'GAMA'] # Insertion


        prev_seq = ''.join([element[flag] for element, flag in zip(self.possible_mutations, self.mutstate)])
        self.mutstate[action] = 1

        # Construct the sequence based on the mutstate flag 
        current_sequence = ''.join([element[flag] for element, flag in zip(self.possible_mutations, self.mutstate)])
        self.current_ids = self.tokenizer(" ".join(current_sequence), return_tensors="pt", padding=True, truncation=True)["input_ids"][0].to(device)


        with torch.no_grad():

            target_mean_embedding, target_attns = get_representation(self.model, self.target_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True)
            current_mean_embedding, current_attns = get_representation(self.model, self.current_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True)

        seqsim = F.cosine_similarity(current_mean_embedding.to(device), target_mean_embedding.to(device)).item()
 

        # Why do I sum here?
        target_attns_arr = np.sum(target_attns.cpu().numpy(), axis=(2, 3))
        current_attns_arr = np.sum(current_attns.cpu().numpy(), axis=(2, 3))
        mse_self = np.mean((target_attns_arr - current_attns_arr) ** 2)
      


        int_limits = [9, 109]

        spacer_len = 100

        spacer = torch.full((1, spacer_len), 5).to(device)
        target_int_ids = torch.cat((self.target_ids.unsqueeze(0),  spacer, int_ids), dim=1)

 
        current_int_ids = torch.cat((self.current_ids.unsqueeze(0), spacer, int_ids), dim=1)
        # target_int_attns are the interface attentions that the binder has

        with torch.no_grad():
            target_int_embeddings, target_int_attns = get_representation(model, target_int_ids, "t5", layers = [-1], output_attentions = True)   
            current_int_embeddings, current_int_attns = get_representation(model, current_int_ids, "t5", layers = [-1], output_attentions = True)




        target_int_attns_trim = trim_attns(attns = target_int_attns, query_int_ids = target_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
        current_int_attns_trim = trim_attns(attns = current_int_attns, query_int_ids = current_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)

        aggregated_target_arr = np.sum(target_int_attns_trim.cpu().numpy(), axis=(2, 3))
        aggregated_current_arr = np.sum(current_int_attns_trim.cpu().numpy(), axis=(2, 3))


        mse_int = np.mean((aggregated_target_arr - aggregated_orig_arr) ** 2)
        print("mse", mse_int)

       
        print("best seqsim", self.best_seqsim, "mse_int", mse_int, "mse_self", mse_self)

        # Better     
        if mse_int < self.lowest_mse_int:
           reward_mse_int = self.lowest_mse_int - mse_int
           self.lowest_mse_int = mse_int
        # Worse
        else:
           reward_mse_int = 0
        print(reward_mse_int)

        # Better     
        if mse_self < self.lowest_mse_self:
           reward_mse_self = self.lowest_mse_self - mse_self
           self.lowest_mse_self = mse_self
        # Worse
        else:
           reward_mse_self = 0
        print(reward_mse_self)



        reward_ss = seqsim - self.best_seqsim

        # Only get a reward for improving on the cosine similarity
        
        # Worse
        if reward_ss <= 0:
            reward_ss = 0

        # Better
        else:
            self.best_seqsim = seqsim
      
        print("REWARDS", reward_ss, reward_mse_int, reward_mse_self)
        #reward = reward_ss
        #reward = reward_mse_int * reward_ss # What if only reward if both drop in mse and increase in ss? Better than ss alone
        reward = reward_mse_int * reward_mse_self  * reward_ss
        self.latest_reward = reward # not using this  
        self.mutation_counter += 1  # Increment mutation counter
        done = self.is_episode_done()

        return self.current_ids, reward, self.mutstate, self.best_seqsim, self.lowest_mse_self, self.lowest_mse_int, done, {}



    def pass_embeddings(self, input_ids, target_ids, model):


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

            target_mean_embedding, target_attns = get_representation(model, target_ids.unsqueeze(0), "t5", layers = [-1]).to(device)
            orig_mean_embedding, orig_attns = get_representation(model, current_ids.unsqueeze(0), "t5", layers = [-1]).to(device)


        cossim_orig_target = F.cosine_similarity(orig_mean_embedding, target_mean_embedding).item()
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
        self.mutstate = mutstate = [0] * len(self.possible_mutations) 
        self.current_ids = self.initial_ids.clone().detach()
        self.mutation_counter = 0  # Reset the mutation counter
        self.latest_reward = 0 
        self.lowest_mse_self = self.starting_mse_self
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



class ActionValueTracker:

    def __init__(self):
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





def parse_arguments():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Optimize protein sequence embeddings with a model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the pretrained model.")

    parser.add_argument("-f", "--fasta_file", type=str, required=True, help="Path to the FASTA file with two sequences: the query and the target.")

    parser.add_argument("-i", "--int_fasta_file", type=str, required=False, help="A fasta file with a single target binding sequence")



    parser.add_argument("-a", "--aln_file", type=str, help="Path to a FASTA alignment file with two sequences: the query and the target.")

    parser.add_argument("-s", "--steps", type=int, default = 100, help="Number of mutations to make.")

    parser.add_argument("-c", "--cnn_path", type=str, required=False, help="Path to disorder predictor model, SETH_CNN.pt")

    parser.add_argument("-e", "--eval", type=str, required=False, help="type of gradient evaluation. default: intgrad_rolling", default = "intgrad_rolling")

    

    # Your previous arguments remain unchanged here...



    return parser.parse_args()

def get_attn_data(model, tokenizer, tokens, min_attn = 0.1,  max_seq_len=1024, model_type = "bert"):





    #if max_seq_len:

    #    tokens = tokens[:max_seq_len - 2]  # Account for SEP, CLS tokens (added in next step)



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





def get_representation(model, input_ids, model_type, layers, mask = None, output_attentions = False):

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

def trim_attns(attns, query_int_ids, int_ids,  spacer_len = 0, int_limits = []):

    offset = query_int_ids.shape[1] - int_ids.shape[1]
    #print("offset", offset, spacer.shape[1], query_ids.shape[1])
    if not int_limits:
        #print(query_int_ids.shape[1], offset, int_ids.shape[1])
        int_limits  = [query_int_ids.shape[1] - int_ids.shape[1], query_int_ids.shape[1]]

    int_limits_offset = [x + offset for x in int_limits]


    assert torch.equal(int_ids[:, int_limits[0]: int_limits[1] + 1], query_int_ids[:, int_limits_offset[0]: int_limits_offset[1] + 1]), "something wrong with int_limits"

    attns_trim = attns[:, :, int_limits_offset[0]: int_limits_offset[1] + 1, int_limits_offset[0] : int_limits_offset[1] + 1]
       
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
    steps = args.steps
    evaluation = args.eval

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

        outfile_base = f"{fasta_file}_{evaluation}"

        orig_seqrecord = sequences[0]
        target_seqrecord = sequences[1]

        orig_name = orig_seqrecord.id
        target_name = target_seqrecord.id

        orig_seq_nospaces = (str(orig_seqrecord.seq))
        target_seq_nospaces = (str(target_seqrecord.seq))

        starting_alignment = do_align_new([target_seqrecord, orig_seqrecord], outfile_base)
        alignment = starting_alignment
    print("orig_seqrecord", orig_seqrecord)
    print("target_seqrecord", target_seqrecord)

    if int_fasta_file:
        with open(int_fasta_file, "r") as handle:
            int_seqrecord = list(SeqIO.parse(handle, "fasta"))[0]
            int_seq_nospaces = str(int_seqrecord.seq)
            int_name = int_seqrecord.id
            int_tokens = tokenizer(" ".join(int_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
            int_ids = int_tokens["input_ids"].to(device)

        int_seq_limits = [0,50]

       


    tokens_orig = tokenizer(" ".join(orig_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
    orig_ids = tokens_orig["input_ids"].to(device)
    tokens_target = tokenizer(" ".join(target_seq_nospaces), return_tensors="pt", padding=True, truncation=True)
    target_ids = tokens_target["input_ids"].to(device)

    print("orig_ids", orig_ids)
    print("target_ids", target_ids)

 
    with torch.no_grad():
    
        target_mean_embedding, target_attns = get_representation(model, target_ids, model_type, layers, output_attentions = True)
        orig_mean_embedding, orig_attns = get_representation(model, orig_ids, model_type, layers, output_attentions = True) 

        spacer_len = 100
        spacer = torch.full((1, spacer_len), 5).to(device)
        target_int_ids = torch.cat((target_ids, spacer, int_ids), dim=1)
        print(target_ids.shape, int_ids.shape, target_int_ids.shape)
        # target_int_attns are the interface attentions that the binder has
        target_int_embeddings, target_int_attns = get_representation(model, target_int_ids, model_type, layers, output_attentions = True)
 
        orig_int_ids = torch.cat((orig_ids, spacer, int_ids), dim=1)
        orig_int_embeddings, orig_int_attns = get_representation(model, orig_int_ids, model_type, layers, output_attentions = True)
     

    target_attns_arr = np.sum(target_attns.cpu().numpy(), axis=(2, 3))
    orig_attns_arr = np.sum(orig_attns.cpu().numpy(), axis=(2, 3))
    mse_self = np.mean((target_attns_arr - orig_attns_arr) ** 2)
 

    cossim_orig_target = F.cosine_similarity(orig_mean_embedding, target_mean_embedding)
    int_limits = [9, 109]



    target_int_attns_trim = trim_attns(attns = target_int_attns, query_int_ids = target_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
    print(target_int_attns_trim.shape)

    total_target_int_attns = get_total_attn(target_int_attns_trim)
    print(total_target_int_attns)

    orig_int_attns_trim = trim_attns(attns = orig_int_attns, query_int_ids = orig_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
    print(orig_int_attns_trim.shape)


    total_orig_int_attns = get_total_attn(orig_int_attns_trim)
    print(total_orig_int_attns)



    # Aggregate the arrays by summing over the last two dimensions
    print(target_int_attns.shape) 

    # Look at nework difference, by only for amino acid positions that are in both query and target.
    

    aggregated_arr1 = np.sum(target_int_attns_trim.cpu().numpy(), axis=(2, 3))
    
    aggregated_arr2 = np.sum(orig_int_attns_trim.cpu().numpy(), axis=(2, 3))
    
    diff = aggregated_arr1 - aggregated_arr2 
    mse_int = np.mean((aggregated_arr1 - aggregated_arr2) ** 2)     

 
    # Plotting
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
   
     
    
    
    # Heatmap for the first aggregated array
    
    axes[0].imshow(diff, cmap='coolwarm', vmin = -20, vmax= 20,  interpolation='nearest')
    
    axes[0].set_title('Aggregated Array 1')
    
    axes[0].set_xlabel('Dimension 2 (size 16)')
    
    axes[0].set_ylabel('Dimension 1 (size 32)')
    
    
    
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
    plt.show() 
    plt.savefig(f'{fasta_file}_aggregated_arrays.png', dpi=300)



    #exit(1)





    #total_target_int_attns = get_total_attn(attns = target_int_attns, query_int_ids = target_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)

    #total_orig_int_attns = get_total_attn(attns = orig_int_attns, query_int_ids = orig_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)


    


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

    mutstate = [0] * len(possible_mutations)
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




    env = ProteinMutationEnv(orig_ids.squeeze(), target_ids.squeeze(), possible_mutations, mutstate, cossim_orig_target.item(), mse_self, mse_int, model, tokenizer, device)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n


    tracker = ActionValueTracker()
    
    # Hyperparameters
    
    learning_rate = 0.1
    gamma = 0.99  # Discount factor
    epsilon = 0.1  # Exploration rate
    epsilon_min = 0.5
    epsilon_decay = 0.995
    num_episodes = 50
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
    for episode in range(num_episodes):
        print("Start episode, ", episode)
        print(tracker.print_cumulative_rewards())
         

 
        counter = 0
        mutstate = env.reset()
        mutstate = torch.tensor(mutstate, dtype=torch.float32).clone().detach().to(device)
        done = False
   
        current_action_space = list(range(env.action_space.n))
        print("current action space", current_action_space)
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
                #print("action", action)
            #else:
            #    action_values = tracker.get_action_values()
            #    action = max(action_values, key=action_values.get)
            #print("previous actions", previous_actions)
            if action not in previous_actions:
                valid_action_found = True
                previous_actions.append(action)


    
          # Take action and observe reward and next state
          next_ids, reward, mutstate, best_seqsim, lowest_mse_self, lowest_mse_int, done, _ = env.step(action)
          next_state = mutstate
          cumulative_rewards_df = tracker.cumulative_rewards_table()
          if len(current_action_space) == 0:
              done = True

          next_state = torch.tensor(next_state, dtype=torch.float32)

          next_state = next_state.clone().detach()  # Added clone detach after warning
          tracker.update(action=action, reward=reward, episode = episode, step = counter)
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
  
          if counter ==0:
                  q_network.eval()
                  with torch.no_grad():
                      q_values_check = q_network(torch.tensor(mutstate, dtype=torch.float32).to(device)) 
                  q_network.train()
                  df_mutannot['q'] = q_values_check.clone().detach().cpu() # detach necessary or not?
                  #print(cumulative_rewards_df)
                  print(df_mutannot.merge(cumulative_rewards_df, how = "left", on = ["action", "aa1"]).sort_values(by='cumulative_reward', ascending=False).head(100))
   
          counter = counter + 1    
      
      
          # Update epsilon
      
          epsilon = max(epsilon_min, epsilon_decay * epsilon)
          # Logging
     
            
          #if episode % 1 == 0:
          #    print(f"Episode: {episode}, Loss: {loss_q.item()}, Epsilon: {epsilon}")
            
    
    
    # Close the environment




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
    total_rewards.to_csv('action_rankings_per_total.csv', index=False)


    # Rank the actions within each episode
    #step_log['rank'] = step_log.groupby('episode')['reward'].rank(method='dense', ascending=False)
    
    #step_log.to_csv('action_rankings_per_episode.csv', index=False)
    
   
    
    
    def apply_actions_in_order(actions, possible_mutations, target_mean_embedding, model, tokenizer, device, int_ids = torch.tensor([]), spacer_len = 0, int_limits = []):
    
        seqsims = []    
        self_attns = []
        int_attns = []
        mutstate = [0] * len(possible_mutations)
        spacer = torch.full((1, spacer_len), 5).to(device)
        for action in actions:
    
            mutstate[action] = 1

            # Construct the sequence based on the mutstate flag 
            current_sequence = ''.join([element[flag] for element, flag in zip(possible_mutations, mutstate)])
 
            current_ids = tokenizer(" ".join(current_sequence), return_tensors="pt", padding=True, truncation=True)["input_ids"][0].to(device)


            with torch.no_grad():

                current_mean_embedding, current_attns  = get_representation(model, current_ids.unsqueeze(0), "t5", layers = [-1], output_attentions = True).to(device)

                aggregated_current_arr = np.sum(current_attns.cpu().numpy(), axis=(2, 3))

                mse_self = np.mean((aggregated_arr1 - aggregated_self_arr) ** 2)
                self_attns.append(mse_self)


                if int_ids.size(0) > 0:
                    current_int_ids = torch.cat((current_ids.unsqueeze(0), spacer, int_ids), dim=1).to(device)

                    current_int_embeddings, current_int_attns = get_representation(model, current_int_ids, "t5", layers = [-1], output_attentions = True)

                    current_int_attns_trim = trim_attns(attns = current_int_attns, query_int_ids = current_int_ids, int_ids = int_ids, spacer_len = spacer_len, int_limits = int_limits)
                    aggregated_int_arr = np.sum(current_int_attns_trim.cpu().numpy(), axis=(2, 3))

                    mse_int = np.mean((aggregated_arr1 - aggregated_int_arr) ** 2)
                    int_attns.append(mse_int)





            seqsim = F.cosine_similarity(current_mean_embedding, target_mean_embedding.to(device)).item()
            seqsims.append(seqsim)
        
        if int_ids.size(0) > 0:
            return seqsims, self_attns, int_attns
        else:
            return seqsims, self_attns
    
    
    
    # Add an empty column for similarity scores
    
    total_rewards['similarity_score'] = None
    
    # Get a list of episodes
    
    episodes = step_log['episode'].unique()
    
    
    for episode in episodes:
    
        # Get actions for this episode sorted by rank
        episode_actions = total_rewards[total_rewards['episode'] == episode]
        #episode_actions= episode_actions.sort_values(by='rank')
        actions_list = episode_actions['action'].tolist()
    
    
        # Apply actions in order
        if int_ids.size(0) > 0:
            similarity_score, attn_score = apply_actions_in_order(actions_list, possible_mutations, target_mean_embedding, model, tokenizer, device, int_ids = int_ids, spacer_len = 100, int_limits = int_limits) 
            total_rewards.loc[total_rewards['episode'] == episode, 'attn_score'] = attn_score
 
        else:
           similarity_score = apply_actions_in_order(actions_list, possible_mutations, target_mean_embedding, model, tokenizer, device) 
    
        # Store the similarity score in the DataFrame
        total_rewards.loc[total_rewards['episode'] == episode, 'similarity_score'] = similarity_score
    
    
    # Display the resulting DataFrame
    print("total rewards", total_rewards)

 
    # Set up plot
    num_episodes = len(episodes)
    cmap = plt.cm.viridis  # You can change 'viridis' to any other colormap
    colors = cmap(np.linspace(0, 1, num_episodes))

    plt.figure(figsize=(10, 6))
    
    
    for idx, episode in enumerate(episodes):
    
        # Filter the data for the current episode
        episode_data = total_rewards[total_rewards['episode'] == episode]
    
        
    
        # Plot the data
        plt.plot(episode_data['rank'], episode_data['similarity_score'], label=episode, color = colors[idx])
    
    
    
    plt.xlabel('Rank')
    plt.ylabel('Similarity score')
    plt.title('Similarity vs Ranked mutation for each Episode')
    plt.show()
    plt.savefig(f'{fasta_file}_similarity_score_plot.png')

    # Optionally, you can save this DataFrame to a CSV file
    
    total_rewards.to_csv(f'{fasta_file}_action_rankings_total_with_similarity.csv', index=False)
    
    ######################################

    if int_ids.size(0) > 0:

        plt.figure(figsize=(10, 6))
    
    
        for idx, episode in enumerate(episodes):
    
            # Filter the data for the current episode
            episode_data = total_rewards[total_rewards['episode'] == episode]
    
        
    
            # Plot the data
            plt.plot(episode_data['rank'], episode_data['attn_score'], color = colors[idx], label=episode)
        
        
        
            plt.xlabel('Rank')
            plt.ylabel('MSE')
            plt.title('MSE vs Ranked mutation for each Episode')
            plt.show()
            plt.savefig(f'{fasta_file}_attn_mse_score_plot.png')
    
            total_rewards.to_csv(f'{fasta_file}_action_rankings_total_with_attnmse.csv', index=False)
    

        plt.figure(figsize=(10, 6))
    
    
        for idx, episode in enumerate(episodes):
    
            # Filter the data for the current episode
            episode_data = total_rewards[total_rewards['episode'] == episode]
    
        
    
            # Plot the data
            plt.plot(episode_data['attn_score'], episode_data['similarity_score'], color = colors[idx], label=episode)
        
        
        
            plt.xlabel('Rank')
            plt.ylabel('MSE')
            plt.title('attn_score vs. similarity_score for each Episode')
            plt.show()
            plt.savefig(f'{fasta_file}_mse_sim_score_plot.png')
 


 
    #############################






















    total_rewards['similarity_score'] = None

    # What about by the similarity score using the random steps?
    total_rewards= total_rewards.sort_values(by='step')
  
    for episode in episodes:
    
        # Get actions for this episode sorted by rank
        episode_actions = total_rewards[total_rewards['episode'] == episode]
        print("episode actions", episode_actions)
        actions_list = episode_actions['action'].tolist()
    
    
        # Apply actions in order
        similarity_score = apply_actions_in_order(actions_list, possible_mutations, target_mean_embedding, model, tokenizer, device) 
    
        # Store the similarity score in the DataFrame
        total_rewards.loc[total_rewards['episode'] == episode, 'similarity_score'] = similarity_score
    
    
    # Display the resulting DataFrame
    print("total rewards", total_rewards)

 
    # Set up plot
    num_episodes = len(episodes)
    cmap = plt.cm.viridis  # You can change 'viridis' to any other colormap
    colors = cmap(np.linspace(0, 1, num_episodes))

    plt.figure(figsize=(10, 6))
    
    
    for idx, episode in enumerate(episodes):
    
        # Filter the data for the current episode
        episode_data = total_rewards[total_rewards['episode'] == episode]
    
        
    
        # Plot the data
        plt.plot(episode_data['step'], episode_data['similarity_score'], label=episode, color = colors[idx])
    
    
    
    plt.xlabel('Random step')
    plt.ylabel('Similarity score')
    plt.title('Similarity vs Random mutation for each Episode')
    plt.show()
    plt.savefig(f'{fasta_file}_similarity_score_plot_random.png')

    
    total_rewards.to_csv(f'{fasta_file}_action_rankings_total_with_similarity_random.csv', index=False)
    






    
    env.close()
    exit(1)
    
