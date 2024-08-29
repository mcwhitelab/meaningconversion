
#  starging alignment = target, queryimport argparse
from Bio import SeqIO, AlignIO
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
import torch
import torch.nn.functional as F
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile
import subprocess
#from Bio.Application import AbstractCommandline
from io import StringIO
from Bio.Align import MultipleSeqAlignment
#from Bio import AlignIO, SeqIO
import argparse
import numpy as np
from SETH_1 import CNN, get_predictions, write_predictions
import pandas as pd

def load_cnn(cnn_path):

    print("Loading SETH_1...")
    predictor = CNN(1, 1024)
    # Load the model directly without checking online

    state = torch.load(cnn_path)
    predictor.load_state_dict(state['state_dict'])

    #if not device.type == 'cpu':
    #    predictor = predictor.half()  # run in half-precision

    predictor = predictor.to(device)
    predictor = predictor.eval()

    return predictor



def get_diso(seq_dict, model, tokenizer, CNN, form = "Cs"):
    #cnn_pat
    #izer = get_prott5(root_dir)

    #CNN = load_CNN_ckeckpoint(root_dir)

    predictions = get_predictions(seq_dict, model, tokenizer, CNN, form)
    #/print(predictions)
    #write_predictions(out_path, predictions, form)
    return(predictions)

def get_disorder_masks(alignment, diso_dict, orig_id, target_id):
    #disorder_seq1 = [1, 0, 0, 0, 1]
    #disorder_seq2 = [0, 1, 0, 0]

    disorder_seq1 = diso_dict[orig_id][1]
    disorder_seq2 = diso_dict[target_id][1]
    #print("orig_id", orig_id)
    #print("target_id", target_id)


 
    # Create new disorder lists based on the alignment
    mod_disorder_seq1 = []
    mod_disorder_seq2 = []


    # The first sequence in the alignment is the target
    #for seq in alignment:
    #    print(seq.id)
    #print(len(disorder_seq1))
    #print(len(alignment[0]))
    #print(len(alignment[0].seq.replace("-", "")))
    #Target
    counter = 0
    for res in alignment[0]:
       #print(res)
       if res == "-":
           mod_disorder_seq2.append("-")
       else:
           #print(counter, disorder_seq1[counter])
           mod_disorder_seq2.append(disorder_seq2[counter])
           counter = counter + 1

    # Orig
    counter = 0
    for res in alignment[1]:
       if res == "-":
           mod_disorder_seq1.append("-")
       else:
           mod_disorder_seq1.append(disorder_seq1[counter])
           counter = counter + 1

    print("mod_seq1", mod_disorder_seq1)
    print("mod_seq2", mod_disorder_seq2)

    out_seq1 = []
    out_seq2 = []

    

    for i, (res1, res2)  in enumerate(zip(mod_disorder_seq1, mod_disorder_seq2)):

        # If one of the sequences has a gap at that position
        if res1 == "-" and res2 != "-":
            #out_seq2.append(res2)
            out_seq2.append(1)
        elif res2 == "-" and res1 != "-":
            #out_seq1.append(res1)
            out_seq1.append(1) 
        # If either sequences has disorder at that position
        elif res1 == 1 or res2 == 1:
            out_seq1.append(1)
            out_seq2.append(1)
        else:
            out_seq1.append(res1)
            out_seq2.append(res2)
    print("out_seq1", out_seq1)
    print("out_seq2", out_seq2)
        
    
    return(out_seq1, out_seq2, mod_disorder_seq1, mod_disorder_seq2)

def clustal_print(alignment):
    # Create an in-memory text stream
    output = StringIO()

    # Write the alignment to the text stream in clustal format
    AlignIO.write(alignment, output, "clustal")
    # Move to the beginning of the text stream
    output.seek(0)

    # Print the formatted alignment
    print(output.read())


def parse_arguments():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Optimize protein sequence embeddings with a model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the pretrained model.")

    parser.add_argument("-f", "--fasta_file", type=str,  help="Path to a FASTA file with two sequences: the query and the target.")

    parser.add_argument("-a", "--aln_file", type=str, help="Path to a FASTA alignment file with two sequences: the query and the target.")

    parser.add_argument("-s", "--steps", type=int, default = 100, help="Number of mutations to make.")

    parser.add_argument("-c", "--cnn_path", type=str, required=True, help="Path to disorder predictor model, SETH_CNN.pt")
  
    parser.add_argument("-e", "--eval", type=str, required=False, help="type of gradient evaluation. default: intgrad_rolling", default = "intgrad_rolling")
  
    

    # Your previous arguments remain unchanged here...



    return parser.parse_args()

def integrated_gradients_embed(input_embed, baseline_ids, model, target_ids, steps=10):



    with torch.no_grad():
       baseline = model.get_input_embeddings()(baseline_ids)

    # Calculate the path from baseline to input

    delta = input_embed - baseline
    #print(input_embed.shape, baseline.shape)

    results = []



    for step in range(steps):

        alpha = step / steps
        interpolated_input = baseline + alpha * delta

        interpolated_input = interpolated_input.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=interpolated_input, labels=target_ids)

        loss = outputs.loss
        loss.backward()
        results.append(interpolated_input.grad.detach().clone())
        interpolated_input.grad.zero_()



    grads = torch.stack(results).mean(0)



    return grads


def convert_pos_to_truepos(seq_aln, pos):
    """
    Convert the position in a sequence with dashes to its position in the sequence without dashes.
    
    Parameters:
    - seq_aln (str): The sequence with dashes.
    - pos (int): The position in the sequence with dashes.
    Returns:
    - int: The position in the sequence without dashes.
    """
    # Ensure pos is valid
    if pos < 0 or pos > len(seq_aln):
        raise ValueError("Position is outside the range of the sequence.")
    # Adjust for 0-indexing
    pos -= 1
    if seq_aln[pos] == '-':
        return None
    # Count non-dash characters up to the given position
    true_pos = 0
    for i in range(pos):
        if seq_aln[i] != '-':
            true_pos += 1
    # If the character at the given position is not a dash, increment the count
    if seq_aln[pos] != '-':
        true_pos += 1
    return true_pos


def pass_embeddings(input_ids, target_ids, model):
    # Convert ids to embeddings
    with torch.no_grad():
       embeddings = model.get_input_embeddings()(input_ids)

    
    # Make embeddings require gradients
    embeddings = embeddings.clone().detach().requires_grad_(True)
    print("embedded")
    # Forward pass with embeddings
    outputs = model(inputs_embeds=embeddings, labels=target_ids)
    print("passed")
    loss = outputs.loss

    # Backward pass to get gradients
    loss.backward()
    return(embeddings)



def get_importance_grad_x_input(input_ids, target_ids, model, tokenizer, input_diso_mask  = None, target_diso_mask = None):

    #print(input_ids.shape)
    #print(len(input_diso_mask))
    #print(len(input_ids))
    # Do I replace the disorder ids with the X character or not?/
    #print(input_ids)
    # 23 is X
    # 127 is <extra_id_0>
    # 5 is G
    rep_value = 23
    #rep_value = 5
    if input_diso_mask is not None:
        input_mask = torch.tensor(input_diso_mask, dtype = torch.bool)
        false_tensor = torch.tensor([False], dtype = torch.bool)
        #print(input_mask.shape)
        #print(false_tensor.shape)
        input_mask = torch.cat((input_mask, false_tensor), dim = 0).unsqueeze(0)
        
        input_ids[input_mask] = rep_value
        #input_ids = [23 if m else v for m, v in zip(input_diso_mask, input_ids)]

    if target_diso_mask is not None:
        target_mask = torch.tensor(target_diso_mask, dtype = torch.bool)
        target_mask = torch.cat((target_mask, false_tensor), dim = 0).unsqueeze(0)
        target_ids[target_mask] = rep_value
    #print("input_ids at get_importance", input_ids)
    #print("target_ids at get_importance", target_ids)


    embeddings = pass_embeddings(input_ids, target_ids, model)
    print("got gradients")
    # Gradients for embeddings
    embedding_gradients = embeddings.grad

    #print("embeddings", embeddings) 
    #print("gradients", embedding_gradients)
    

    # f. Gradient x Input
    importance_grad_input = (embedding_gradients * embeddings).sum(dim=-1).squeeze() 
    imps = importance_grad_input.detach().tolist()


    ##
    importance_grad_input_max, _ = (embedding_gradients * embeddings).max(dim=-1) 
    # both neg or both pos will end up pos. 
    imps_max = importance_grad_input_max.squeeze().detach().tolist()

    ##
#
    #/print(embedding_gradients)
    importance_norm = (embedding_gradients).norm(p = 2, dim = 2).squeeze() #sum(dim = -1).squeeze()
    #print(importance_norm)
    imps_norm = importance_norm.detach().tolist()

    
    importance_grad = (embedding_gradients).sum(dim = -1).squeeze()
    imps_grad = importance_grad.detach().tolist()


    ###
    product = embedding_gradients * embeddings
    positive_mask = product > 0

    positive_product = product * positive_mask.float()

    ###
    importance_grad_input_pos = positive_product.sum(dim=-1).squeeze()
    imps_pos = importance_grad_input_pos.detach().tolist()



    baseline_input = torch.zeros_like(input_ids).int()
    int_grad = integrated_gradients_embed(embeddings, baseline_input, model, target_ids)


    imps_intgrad = torch.norm(int_grad, dim=-1).squeeze().detach().tolist()




    imps_intgrad_input = (int_grad * embeddings).sum(dim=-1).squeeze().detach().tolist()

    ###
    # Create a mask where both values are positive
    positive_mask = (embedding_gradients > 0) & (embeddings > 0)
    # Compute the product
    product = embedding_gradients * embeddings
    # Use the mask to set non-positive values to zero
    product[~positive_mask] = 0

    # Compute the sum along the desired dimension
    sum_values = product.sum(dim=-1).squeeze()
    imps_bothpos = sum_values.detach().tolist()

    ###
    # Create a mask where both values are negative
    negative_mask = (embedding_gradients < 0) & (embeddings < 0)
    # Compute the product
    product = embedding_gradients * embeddings
    # Use the mask to set non-positive values to zero
    product[~negative_mask] = 0

    # Compute the sum along the desired dimension
    sum_values = product.sum(dim=-1).squeeze()
    imps_bothneg = sum_values.detach().tolist()


    # Create a mask where both values are positive
    negpos_mask = (embedding_gradients < 0) & (embeddings > 0)
    # Compute the product
    product = embedding_gradients * embeddings
    # Use the mask to set non-positive values to zero
    product[~negpos_mask] = 0

    # Compute the sum along the desired dimension
    sum_values = product.sum(dim=-1).squeeze()
    imps_negpos = sum_values.detach().tolist()

    # Create a mask where both values are positive
    posneg_mask = (embedding_gradients > 0) & (embeddings < 0)
    # Compute the product
    product = embedding_gradients * embeddings
    # Use the mask to set non-positive values to zero
    product[~posneg_mask] = 0

    # Compute the sum along the desired dimension
    sum_values = product.sum(dim=-1).squeeze()
    imps_posneg = sum_values.detach().tolist()

 

    pos =  list(range(len(imps)))
    decoded = [tokenizer.decode(x) for x in input_ids[0]]
    decoded_targ = [tokenizer.decode(x) for x in target_ids[0]]

    

    df = pd.DataFrame({
    
        'pos': pos,
    
        'decoded': decoded,
    
        'imps': imps,
    
        'imps_norm': imps_norm,
    
        'imps_max': imps_max,
    
        'imps_grad': imps_grad,
    
        'imps_pos': imps_pos,
    
        'imps_bothpos': imps_bothpos,
    
        'imps_bothneg': imps_bothneg,
    
        'imps_intgrad': imps_intgrad,
    
        'imps_intgrad_input': imps_intgrad_input,
    
        'imps_posneg': imps_posneg,
    
        'imps_negpos': imps_negpos
    
    })
    
    
    window_size = 5  # Adjust this based on your requirements



    # Iterate over each column in the DataFrame, compute rolling mean, and store in a new column

    for column in df.columns:
        if column != 'pos':  
            if column != 'decoded':
                df[f'{column}_rolling'] = df[column].rolling(window=window_size).mean()
    




    #print("imps", imps)
    #sorted_pairs = sorted(zip(tokenizer.decode(x) for x in input_ids[0]], importance_grad_input), key=lambda x: x[1])
    #print("decoded", decoded)
    #print("decoded_targ", decoded_targ)
    #pairs = list(zip(pos, decoded, imps, imps_norm, imps_max, imps_grad, imps_pos, imps_bothpos, imps_bothneg, imps_intgrad, imps_intgrad_input, imps_posneg, imps_negpos))

    return(df)


def do_align(alignment, seqrecordlist, filename):

    #alignment = ...  # Some AlignIO object
    #new_sequences = ...  # Some SeqIO object or list of SeqRecord objects
     
    # Convert AlignIO object and SeqRecord list to FASTA formatted strings
    
    for new_seq in seqrecordlist:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta.aln") as temp_aln:
            AlignIO.write(alignment, temp_aln, "fasta")
            temp_aln_name = temp_aln.name 
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as temp_seq:
            SeqIO.write([new_seq], temp_seq, "fasta")
            temp_seq_name = temp_seq.name
        output_file = f"{filename}.tmp.aln"
        # Set up MUSCLE command line for profile-profile alignment
        cmd = [

          "mafft",
 
          "--quiet", 
 
          "--localpair", 
 
          "--add", temp_seq_name, 

          temp_aln_name, 

          ">", output_file

        ]

        process = subprocess.run(" ".join(cmd), shell=True, check=True)


        #mafft_cline = AbstractCommandline("mafft")

        #mafft_cline.set_parameter("--add", f"{temp_seq_name} {temp_aln_name}")

        #print(mafft_cline)


        #stdout, stderr = mafft_cline()

        #muscle_cline = MuscleCommandline("muscle", profile=True, in1 = temp_aln_name, in2 = temp_seq_name)
        #stdout, stderr = muscle_cline() 
        
        # Convert the stdout string to an AlignIO object
        alignment = AlignIO.read(output_file, "fasta")
        print("added alignment", alignment)
    # Now, `output_alignment` is an AlignIO object with the combined alignment.
    return(alignment) 



def do_align_new(seqrecordlist, filename):
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as temp_fasta:

        SeqIO.write(seqrecordlist, temp_fasta, "fasta")
        temp_fasta_name = temp_fasta.name
        # Align using Clustal Omega

    clustalo_exe = "clustalo"  # Adjust if necessary

    cline = ClustalOmegaCommandline(clustalo_exe, infile=temp_fasta_name, outfile=f"{filename}.output.aln", verbose=True, auto=True, force = True) # Comput up with outfile based on fasta
    stdout, stderr = cline()
    alignment = AlignIO.read(f"{filename}.output.aln", "fasta")
    #print(alignment)
    return(alignment)

def map_positions(seq1, seq2):

    """
    Create a dictionary mapping positions in sequence 1 to positions in sequence 2.
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
        #print(aa_embeddings.shape)



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


def get_representation(model, input_ids, model_type, layers, mask = None):

    model_output =  model.encoder(input_ids=input_ids)

    aa_embeddings, aa_shape = retrieve_aa_embeddings(model_output, model_type = model_type, layers = layers)
    aa_embeddings = aa_embeddings.to('cpu')
    #aa_embeddings = np.array(aa_embeddings)

   # print(aa_embeddings.shape)
    if mask is not None:
       bool_mask = torch.tensor(mask, dtype=torch.bool)
       masked_embeddings = aa_embeddings[:, ~bool_mask]
       print("masked_embeddings.shape", masked_embeddings.shape)
       sequence_representation = torch.mean(masked_embeddings, dim = 1)
    else:
        sequence_representation = torch.mean(aa_embeddings, dim = 1)#np.mean(aa_embeddings, axis = 1)
    return(sequence_representation)

def generate_variants(seqA, seqB, alignment_dict):

    variants = []
    for key, value in alignment_dict.items():
        # Create a variant by replacing the character at the key position in seqA

        # with the character at the value position in seqB

        variant = seqA[:key] + seqB[value] + seqA[key+1:]
        variants.append([variant, seqA[key], seqB[value], key, value])

    return variants


def replace_char_at_index(orig_string, index, char):
    return orig_string[:index] + char + orig_string[index + 1:]


args = parse_arguments()
fasta_file = args.fasta_file
model_path = args.model_path
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

#exit(1)


# cpu and gpu create slightly different gradients, unknown why
#cpu_only = True
cpu_only = False
if cpu_only == True:
    device = "cpu"
elif torch.cuda.is_available():
    device = torch.device("cuda")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
device_ids =list(range(0, torch.cuda.device_count()))
print("device_ids", device_ids)
model = model.to(device)

# How does the layer choice affect things? Is it only in sequence similarity tests?
#layers = [-2, -1]
#layers = [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
#layers = [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
layers = -1

# Ensure model parameters are not updated
for param in model.parameters():
    param.requires_grad = False



# CHANGE ALL TO THE QUERY -> TARGET ORDER. Otherwise too confusing

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

if aln_file:
    outfile_base = aln_file
    # it's nice to start with two sequences, query,target.
    # but it's convenient afterward to have the target as the first sequence
    starting_alignment = AlignIO.read(aln_file, "fasta")

    print("starting", starting_alignment)
    #starting_alignment[:] = [starting_alignment[1], starting_alignment[0]] 
    starting_alignment = MultipleSeqAlignment([starting_alignment[1], starting_alignment[0]])
    print("starting switch to target then query", starting_alignment)
    alignment = starting_alignment


    orig_seq_wgaps = alignment[1].seq
    target_seq_wgaps = alignment[0].seq
 
    orig_name = alignment[1].id
    target_name = alignment[0].id

    orig_seq_nospaces = (str(orig_seq_wgaps.replace("-", "")))
    target_seq_nospaces = (str(target_seq_wgaps.replace("-", "")))

 
    orig_seqrecord = SeqRecord(Seq(orig_seq_nospaces), id = orig_name)
    target_seqrecord = SeqRecord(Seq(target_seq_nospaces), id = target_name)

    

orig_seq = " ".join(orig_seq_nospaces)
target_seq = " ".join(target_seq_nospaces)
    

print(orig_seq)
print(target_seq)



#seq_record_target = SeqRecord(Seq("".join(target_seq)), id=target_name)
#seq_record_orig= SeqRecord(Seq("".join(orig_seq)), id=orig_name)


# start with target, then input, then modifications of the input. 
# Makes more sense to finish with input, then changes as it approaches the target maybe
output_seqrecords = [target_seqrecord, orig_seqrecord]
orig_ids = tokenizer(orig_seq, return_tensors="pt").input_ids.to(device)


target_ids = tokenizer(target_seq, return_tensors="pt").input_ids.to(device)



#DISORDER MASK
CNN = load_cnn(cnn_path)
print("CNN for disorder prediction loaded")
seq_dict = {target_seqrecord.id:target_seqrecord.seq, orig_seqrecord.id:orig_seqrecord.seq}
print(seq_dict)
diso_dict = get_diso(seq_dict, model.encoder, tokenizer, CNN)

orig_diso_mask, target_diso_mask, orig_diso_mask_aln, target_diso_mask_aln = get_disorder_masks(alignment, diso_dict, orig_name, target_name)
#orig_diso_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#target_diso_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#print(diso_dict[orig_name][1])
#print(diso_dict[target_name][1])
#print(orig_diso_mask)
#print(target_diso_mask)

# DON'T JUST DO THE FINAL LAYER
# JUST DO LAST 16, better representation
with torch.no_grad():
    target_mean_embedding = get_representation(model, target_ids, model_type, layers)
    orig_mean_embedding = get_representation(model, orig_ids, model_type, layers)

    print("before masking disorder", F.cosine_similarity(orig_mean_embedding, target_mean_embedding))
    #print(target_mean_embedding)
    #tmp = target_mean_embedding
    target_mean_embedding = get_representation(model, target_ids, model_type, layers, mask = target_diso_mask)
    orig_mean_embedding = get_representation(model, orig_ids, model_type, layers, mask = orig_diso_mask)
    #print(target_mean_embedding) 
    print("after masking disorder ", F.cosine_similarity(orig_mean_embedding, target_mean_embedding))


current_seq = orig_seq
current_diso_mask = orig_diso_mask
cossim_orig_target = F.cosine_similarity(orig_mean_embedding, target_mean_embedding)
cossim_current_orig_hist = []
cossim_current_target_hist = []

current_ids = tokenizer(current_seq, return_tensors="pt").input_ids.to(device)
#with torch.no_grad():
#    current_mean_embedding = get_representation(model, current_ids, model_type, layers)
current_mean_embedding = orig_mean_embedding

cossim_current_orig = F.cosine_similarity(current_mean_embedding, orig_mean_embedding) 
cossim_current_target =  F.cosine_similarity(current_mean_embedding, target_mean_embedding) 

print("cossim_orig_target :", cossim_orig_target)    
print("cossim_current_orig:", cossim_current_orig)
print("cossim_current_targ:", cossim_current_target)     

cossim_current_orig_hist.append(cossim_current_orig.item())
cossim_current_target_hist.append(cossim_current_target.item())

# First score is to the target, which will have cossim of 1 to itself
# Score score is the original
output_scores = [1, cossim_orig_target.item()]

print(cossim_current_orig_hist)
print(cossim_current_target_hist)

df_target_start = get_importance_grad_x_input(orig_ids, target_ids, model, tokenizer, orig_diso_mask ,target_diso_mask)
df_orig_start = get_importance_grad_x_input(orig_ids, orig_ids, model, tokenizer, orig_diso_mask ,orig_diso_mask)


alignment_map = map_positions(alignment[-1].seq, alignment[0].seq)
variants = generate_variants(orig_seq_nospaces, target_seq_nospaces, alignment_map)


orig_difflist = []
target_difflist = []
poslist = []

orig_cossimlist = []
target_cossimlist = []
for var in variants:
     
    # don't bother when the amino acid is unchaged
    if var[2] == var[1]:
        continue

    # don't make disordered swaps
    if orig_diso_mask[var[3]] == 1:
        continue

    if target_diso_mask[var[4]] == 1:
        continue
 
    print(var)
    var_ids = tokenizer(" ".join(var[0]), return_tensors="pt").input_ids.to(device) 
 
    df_target = get_importance_grad_x_input(var_ids, target_ids, model, tokenizer, orig_diso_mask ,target_diso_mask)
    df_orig = get_importance_grad_x_input(var_ids, orig_ids, model, tokenizer, orig_diso_mask ,orig_diso_mask)

    #print(df_target)
    #print(df_orig) 

    #print(orig_ids)
    #print(var_ids)
    #print(target_ids)

    df_target_diff = df_target_start[df_target_start['decoded'] != "_X"]['imps_norm'][:-1] - df_target[df_target['decoded'] != "_X"]['imps_norm'][:-1]
    df_orig_diff = df_orig_start[df_orig_start['decoded'] != "_X"]['imps_norm'][:-1] - df_orig[df_orig['decoded'] != "_X"]['imps_norm'][:-1]

    #print(df_target_diff.tolist())
    #print(df_orig_diff.tolist())
   
    target_diff_mean = df_target_diff.mean()   
    orig_diff_mean = df_orig_diff.mean()

    #print(target_diff_mean)
    #print(orig_diff_mean) 
    poslist.append(var[3])
    target_difflist.append(target_diff_mean)
    orig_difflist.append(orig_diff_mean)

    with torch.no_grad():
        var_mean_embedding = get_representation(model, var_ids, model_type, layers)

    var_target_cossim =  F.cosine_similarity(var_mean_embedding, target_mean_embedding)
    var_orig_cossim =  F.cosine_similarity(var_mean_embedding, orig_mean_embedding)

    target_cossimlist.append(var_target_cossim)
    orig_cossimlist.append(var_orig_cossim)

  
for x in list(zip(poslist, target_difflist, orig_difflist, target_cossimlist, orig_cossimlist)):
  print(x[0], x[1], x[2], x[3], x[4])


exit(1)


print(alignment)

with open("imps.csv", "w") as f:
  f.write("os,decoded,imps,imps_norm,imps_max,imps_grad,imps_pos,imps_bothpos,imps_bothneg,imps_intgrad,imps_intgrad_input,imps_posneg,imps_negpos\n")
  for step in range(steps):
 
   
    # Example usage
    # get mapping between the latest added sequence and the target sequence
    alignment_map = map_positions(alignment[-1].seq, alignment[0].seq)

   

    print(alignment_map)
    print(target_diso_mask)
    #print("alignment_map", alignment_map)


    # Masking very helpful then there's disordered sequence
    df = get_importance_grad_x_input(current_ids, target_ids, model, tokenizer, orig_diso_mask ,target_diso_mask)
    #pairs = get_importance_grad_x_input(current_ids, target_ids, model, tokenizer)
  
    
    if evaluation == "intgrad_rolling": 
        df = df.sort_values(by='imps_intgrad_rolling', ascending=False)

    if evaluation == "intgradxinput_rolling": 
        df = df.sort_values(by='imps_intgrad_input_rolling', ascending=False)
     
    df[~df['decoded'].str.contains('X')]

    print(df)
    df.to_csv("TEST.csv") 
    #exit(1)
    sorted_pairs = list(zip(df['pos'], df['decoded']))
    #exit(1)
    #for x in pairs:
    #    print(x)
    #print("RANKING")

    # To sort by sum of grad * input
    #sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse = True)
    #sorted_pairs = sorted(pairs, key=lambda x: x[9], reverse = True) # * input
    for x in sorted_pairs:# [:100]:
        print("sorted pair", x)
        f.write(f"{step},{x}\n")
       #if x[1] == '</s>':
       #     print("hey")
          
    print(alignment_map)
    print("targ align", alignment[0].seq)
    print("curr align", alignment[-1].seq) 
    # Switch out amino acids from the target sequence
    print(alignment_map)
    for x in sorted_pairs:
        if x[1] == '</s>':
            continue
        if x[1] == "X":
            continue
        print("start looking at variant")
        print(f"variant: {x}")
        pos_current = x[0]
        if pos_current in alignment_map.keys():
 
            pos_target = alignment_map[pos_current] # find equivalent position in target
            #print("If the target position is disordered, don't swap it to the current sequence")
            #print("is target a disordered point?")
            #print(target_diso_mask)
            #print(target_seqs_nospaces)
            #print(len(target_diso_mask))
            #print(len(target_seqs_nospaces))

            aa_target = target_seq_nospaces[pos_target]
            #print(aa_target)
            #print(target_diso_mask[pos_target])
            target_diso_status = target_diso_mask[pos_target]
        # Potentially just skip if there's a insertion/deletion. But keep for now. 
        else:
            continue
        #else:
        #    aa_target = ""
        #    pos_target = None
        current_seq_nospaces =  current_seq.replace(" ", "")
  
        
 
        print("curr no spaces", current_seq_nospaces)
        #print("len no spaces", len(current_seq_nospaces))     
        print("pos current", pos_current)
      
        #print("minus 1",current_seq_nospaces[pos_current - 1])
        #print("normal ", current_seq_nospaces[pos_current])     
        aa_current = current_seq_nospaces[pos_current]
        current_diso_status = current_diso_mask[pos_current]
        

        print(f"pos_current: {pos_current}, pos_target: {pos_target}, aa_current: {aa_current}, aa_target: {aa_target}, target_diso_status {target_diso_status}, current_diso_status {current_diso_status}")
        print(f"targ seq: {target_seq_nospaces}")
        print(f"orig seq: {orig_seq_nospaces}")
        print(f"prev seq: {current_seq_nospaces}")


          
        if aa_current != aa_target and target_diso_status == 0 and current_diso_status == 0 and aa_target != "-":
              
            # get score
            print(f"replacing {aa_current}-{pos_current} with {aa_target}")
            candidate_seq_nospaces = replace_char_at_index(current_seq_nospaces, pos_current, aa_target)
            # If there's been a deletion
            # Are insertions currently happening?
            #if aa_target == "-":
            #   candidate_diso_mask =  current_diso_mask[:pos_current] + current_diso_mask[pos_current + 1:]

            # If there's been no change in spacing
            #else:
            #   candidate_diso_mask = current_diso_mask
            candidate_diso_mask = orig_diso_mask
            print(f"new seq : {current_seq_nospaces}")

            candidate_seq = " ".join(candidate_seq_nospaces)
            candidate_ids = tokenizer(candidate_seq, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                #candidate_representation = model.encoder(input_ids=candidate_ids)[0]
                #candidate_mean_embedding = torch.mean(candidate_representation, dim=1)
                candidate_mean_embedding = get_representation(model, candidate_ids, model_type, layers, mask = candidate_diso_mask)
      
   
            cossim_candidate_orig = F.cosine_similarity(candidate_mean_embedding, orig_mean_embedding) 
            cossim_candidate_target =  F.cosine_similarity(candidate_mean_embedding, target_mean_embedding) 

            if  cossim_candidate_target > cossim_current_target_hist[-1]: 
                print("KEEPING") 

                print("cossim_orig_target :", cossim_orig_target)    
                print("cossim_candidate_orig:", cossim_candidate_orig)
                print("cossim_candidate_targ:", cossim_candidate_target)     
            
                cossim_current_orig_hist.append(cossim_candidate_orig.item())
                cossim_current_target_hist.append(cossim_candidate_target.item())
        
                print(cossim_current_orig_hist)
                print(cossim_current_target_hist)        
 
                current_seq_nospaces = candidate_seq_nospaces
                current_seq = candidate_seq
                current_ids = candidate_ids
                current_diso_mask = orig_diso_mask
                cossim_value = cossim_candidate_target.item()
                current_seqrecord =  SeqRecord(Seq(current_seq_nospaces), 
   
                       id=f"{orig_name}_{step}_{aa_current}{pos_current + 1}{aa_target}_{cossim_value:.4f}")
                output_seqrecords.append(current_seqrecord)
                output_scores.append(cossim_value)
                # The first two sequences are target and query
                alignment = do_align(alignment, [current_seqrecord], outfile_base) # Add on the latest seqrecord
                clustal_print(alignment)
 
            #current_seq = " ".join(current_seq_nospaces)
                break
            else: 
                print("NOT KEEPING")
                continue

        else:
            continue   


print("cossim_orig_target :", cossim_orig_target)    
print("cossim_current_orig:", cossim_current_orig)
print("cossim_current_targ:", cossim_candidate_target)     
#exit(0)
    
#cossim_current_orig_hist.append(cossim_current_orig)
#cossim_current_target_hist.append(cossim_current_target)

print(cossim_current_orig_hist)
print(cossim_current_target_hist)


with open(f"{outfile_base}_phase1_output.aln", "w") as handle:
     AlignIO.write(alignment, handle, "clustal")

with open(f"{outfile_base}_phase1_output.aln.fasta", "w") as handle:
     AlignIO.write(alignment, handle, "fasta")

with open(f"{outfile_base}_phase1_output.fasta", "w") as handle:
     SeqIO.write(output_seqrecords, handle, "fasta")

   
#exit(1)



# OK step 2. We have a similar sequence. Replace each amino acid back to the original, keeping the cosine similarity high. (Revert as many mutations as possible)
# The better the scoring for step 1, the fewer reversions would need to be made



print(alignment) 

alignment = starting_alignment
print("PHASE 2, revert towards original")

# The first SeqRecord, and the first score is the target sequence, so skip it
combined_data = zip(output_seqrecords[1:], output_scores[1:])
highest_score_seq = max(combined_data, key=lambda x: x[1])[0]

print(output_scores)
print([x.id for x in output_seqrecords])

# Now, highest_score_seq contains the SeqRecord with the highest score
print(alignment[0].seq)
print(alignment[1].seq)
print(highest_score_seq.id)
print(highest_score_seq.seq)
p2_start = highest_score_seq

# Get all three aligned, orig, mod, target
#  starging alignment = target, query, so add on the highest scoring seq at the end
p2_alignment = do_align(alignment, [highest_score_seq], outfile_base)#[target_seqrecord]) #[target_seqrecord, orig_seqrecord, highest_score_seq])
print("p2 alignment", p2_alignment)



#exit(1)
# keep the highest scoring sequence, and do again. 

# currently just showing that there are ones you can revert while keeping the high score


p2_output_list = [target_seqrecord]

mod_history = []
change_record = []


#starting alignment = target, orig, etc. 
orig_seq_aln = p2_alignment[1].seq
target_seq_aln = p2_alignment[0].seq
mod_seq_aln =  p2_alignment[2].seq # Real, just for testing
#mod_seq_aln = target_seq_aln # just for testing

print("orig_seq_aln  ", orig_seq_aln)
print("mod_seq_aln   ", mod_seq_aln)
print("target_seq_aln", target_seq_aln)


candidate_diso_mask = orig_diso_mask

#steps = len(orig_seq_aln)
print("steps", steps)
for i in range(0,steps):

    print(f"ITERATION {i}")
    modified_seqs = []
    # revert each position, and keep the highest scoring sequence after reversion. 
    for pos in range(0,len(orig_seq_aln)):
              

       if orig_seq_aln[pos] == mod_seq_aln[pos]:
            continue
       else:
          #print(pos)
          p2_candidate = "{}{}{}".format(mod_seq_aln[0:pos], orig_seq_aln[pos], mod_seq_aln[pos+1:])
          #if orig_seq_aln[pos] == "-":
          #     candidate_diso_mask = candidate_diso_mask[0:pos] + candidate_diso_mask[pos + 1:]  
          #else:
          #     candidate_diso_mask = candidate_diso_mask        

          with torch.no_grad():
               candidate_ids = tokenizer(" ".join(p2_candidate.replace("-", "")), return_tensors="pt").input_ids.to(device)
               #candidate_representation = model.encoder(input_ids=candidate_ids)[0]
               #candidate_mean_embedding = torch.mean(candidate_representation, dim=1)
               candidate_mean_embedding = get_representation(model, candidate_ids, model_type, layers, candidate_diso_mask)

 
               cossim_candidate_orig = F.cosine_similarity(candidate_mean_embedding, orig_mean_embedding) 
               cossim_candidate_target =  F.cosine_similarity(candidate_mean_embedding, target_mean_embedding) 
               #print(pos, p2_alignment[1].seq[pos], p2_alignment[0].seq[pos])
               #print(cossim_candidate_orig.item(), cossim_candidate_target.item())
               modified_seqs.append((p2_candidate, cossim_candidate_target.item(), pos, mod_seq_aln[pos], orig_seq_aln[pos] ))

    for x in modified_seqs:
          print("mod seq candidate", x)

    if len(modified_seqs) > 0: 
        
        best_score = max(modified_seqs, key=lambda x: x[1])
        # pos, 
        change_record.append([best_score[2], best_score[3], best_score[4]])
        print("best", best_score[2], best_score[3], best_score[4])

        print("current best score", best_score[1])
        print("Best scoring seq", best_score[0])
        highest_score_seq = SeqRecord(Seq(best_score[0]), id=f"del{best_score[2] + 1}_{orig_name}_{best_score[1]:.4f}")
 
        mod_seq_aln = best_score[0]

        # Do we need to keep redoing this alignment?
        # Could it affect the position tracking?
        # Yes
        #p2_alignment = do_align([orig_seqrecord, highest_score_seq, target_seqrecord])
       

        p2_output_list.append(highest_score_seq)
    else:
        print("breaking in phase 2")
        break

for x in change_record:
        print(x)


print("list", p2_output_list)
p2_output_list = p2_output_list + [orig_seqrecord]
print("appe", p2_output_list)
p2_output_alignment = do_align(alignment, p2_output_list, outfile_base)
print(p2_output_alignment)
with open(f"{outfile_base}_phase2_output.aln", "w") as handle:
     AlignIO.write(p2_output_alignment, handle, "clustal")

with open(f"{outfile_base}_phase2_output.aln.fasta", "w") as handle:
     AlignIO.write(p2_output_alignment, handle, "fasta")

with open(f"{outfile_base}_phase2_output.fasta", "w") as handle:
     SeqIO.write(p2_output_list, handle, "fasta")



# Phase 3
# Adding back in mutations, as ranked by phase 2!

#alignment = starting_alignment


mod_seq_aln = orig_seq_aln

candidate_diso_mask = orig_diso_mask

p3_output_list = [orig_seqrecord]

for change in change_record[::-1]:
    pos = change[0]
    #if pos in [96]:
        #  continue

    print("TEST")
    print("mod seq ", mod_seq_aln)
    print("targ seq", target_seq_aln)
    print(pos)
    mod_seq_aln = "{}{}{}".format(mod_seq_aln[0:pos], target_seq_aln[pos], mod_seq_aln[pos+1:])

    #if target_seq_aln[pos] == "-":
    #i    candidate_diso_mask = candidate_diso_mask[0:pos] + target_diso candidate_diso_mask[pos + 1:]  
    #else:
    #    candidate_diso_mask = candidate_diso_mask        


    print("new mod", mod_seq_aln)
    print("modnospace", mod_seq_aln.replace("-", ""))
    true_pos = convert_pos_to_truepos(mod_seq_aln, pos)
    print(f"pos {pos} : true_pos {true_pos}") 


    if true_pos is not None:
       true_pos_output = true_pos + 1
    else:
       true_pos_output = None
    #exit(1)
    with torch.no_grad():
        candidate_ids = tokenizer(" ".join(mod_seq_aln.replace("-", "")), return_tensors="pt").input_ids.to(device)
        #candidate_representation = model.encoder(input_ids=candidate_ids)[0]
        #candidate_mean_embedding = torch.mean(candidate_representation, dim=1)
      
        candidate_mean_embedding = get_representation(model, candidate_ids, model_type, layers, candidate_diso_mask)

        cossim_candidate_orig = F.cosine_similarity(candidate_mean_embedding, orig_mean_embedding) 
        cossim_candidate_target =  F.cosine_similarity(candidate_mean_embedding, target_mean_embedding) 
    print(change, cossim_candidate_orig.item(), cossim_candidate_target.item()) 
    mod_seqrecord = SeqRecord(Seq(mod_seq_aln), id=f"{change[2]}_aln{change[0] + 1}-sq{true_pos_output}_{change[1]}_{orig_name}_{cossim_candidate_target.item():.4f}")
    p3_output_list.append(mod_seqrecord)

p3_output_list.append(target_seqrecord)

p3_output_alignment = do_align(alignment, p3_output_list, outfile_base)
print(p3_output_alignment)
with open(f"{outfile_base}_phase3_output.aln", "w") as handle:
     AlignIO.write(p3_output_alignment, handle, "clustal")

with open(f"{outfile_base}_phase3_output.aln.fasta", "w") as handle:
     AlignIO.write(p3_output_alignment, handle, "fasta")

with open(f"{outfile_base}_phase3_output.fasta", "w") as handle:
     SeqIO.write(p3_output_list, handle, "fasta")


