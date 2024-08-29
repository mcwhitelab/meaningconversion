
#  starging alignment = target, queryimport argparse
from Bio import SeqIO, AlignIO
from transformers import T5Tokenizer, T5ForConditionalGeneration
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

   
    

    # Your previous arguments remain unchanged here...



    return parser.parse_args()


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



def get_importance_grad_x_input(input_ids, target_ids, model, tokenizer):

    embeddings = pass_embeddings(input_ids, target_ids, model)
    print("got gradients")
    # Gradients for embeddings
    embedding_gradients = embeddings.grad

    #print("embeddings", embeddings) 
    #print("gradients", embedding_gradients)
    

    # f. Gradient x Input
    importance_grad_input = (embedding_gradients * embeddings).sum(dim=-1).squeeze()
 
    imps = importance_grad_input.detach().tolist()
    print("imps", imps)
    #sorted_pairs = sorted(zip(tokenizer.decode(x) for x in input_ids[0]], importance_grad_input), key=lambda x: x[1])
    pos =  list(range(len(imps)))
    decoded = [tokenizer.decode(x) for x in input_ids[0]]
    print("decoded", decoded)
    pairs = list(zip(pos, decoded, imps))

    return(pairs)


def do_align(alignment, seqrecordlist):

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
        output_file = "tmp.aln"
        # Set up MUSCLE command line for profile-profile alignment
        cmd = [

          "mafft",
 
          "--quiet", 

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



def do_align_new(seqrecordlist):
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fasta") as temp_fasta:

        SeqIO.write(seqrecordlist, temp_fasta, "fasta")
        temp_fasta_name = temp_fasta.name
        # Align using Clustal Omega

    clustalo_exe = "clustalo"  # Adjust if necessary

    cline = ClustalOmegaCommandline(clustalo_exe, infile=temp_fasta_name, outfile="output.aln", verbose=True, auto=True, force = True) # Comput up with outfile based on fasta
    stdout, stderr = cline()
    alignment = AlignIO.read("output.aln", "fasta")
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


def replace_char_at_index(orig_string, index, char):
    return orig_string[:index] + char + orig_string[index + 1:]


args = parse_arguments()
fasta_file = args.fasta_file
model_path = args.model_path
aln_file = args.aln_file
steps = args.steps

if not fasta_file:
   if not aln_file:
      print("Provide fasta or alignment")
      exit(1)

model = T5ForConditionalGeneration.from_pretrained(model_path).eval()
tokenizer = T5Tokenizer.from_pretrained(model_path)

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


# Ensure model parameters are not updated
for param in model.parameters():
    param.requires_grad = False



# CHANGE ALL TO THE QUERY -> TARGET ORDER. Otherwise too confusing

# Read sequences from FASTA
if fasta_file:
    with open(fasta_file, "r") as handle:
        sequences = list(SeqIO.parse(handle, "fasta"))
 
    outfile_base = fasta_file

    orig_seqrecord = sequences[0]
    target_seqrecord = sequences[1]


    orig_name = orig_seqrecord.id
    target_name = target_seqrecord.id

    orig_seq_nospaces = (str(orig_seqrecord.seq))
    target_seq_nospaces = (str(target_seqrecord.seq))

    starting_alignment = do_align_new([target_seqrecord, orig_seqrecord])
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

with torch.no_grad():

    target_representation = model.encoder(input_ids=target_ids)[0]
    target_mean_embedding = torch.mean(target_representation, dim=1)


with torch.no_grad():
    orig_representation = model.encoder(input_ids=orig_ids)[0]
    orig_mean_embedding = torch.mean(orig_representation, dim=1)


current_seq = orig_seq

cossim_orig_target = F.cosine_similarity(orig_mean_embedding, target_mean_embedding)
cossim_current_orig_hist = []
cossim_current_target_hist = []

current_ids = tokenizer(current_seq, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    current_representation = model.encoder(input_ids=current_ids)[0]
    current_mean_embedding = torch.mean(current_representation, dim=1)

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



print(alignment)

for step in range(steps):
 
   
    # Example usage
    # get mapping between the latest added sequence and the target sequence
    alignment_map = map_positions(alignment[-1].seq, alignment[0].seq)

    #print("alignment_map", alignment_map)


    pairs = get_importance_grad_x_input(current_ids, target_ids, model, tokenizer)

    #for x in pairs:
    #    print(x)
    #print("RANKING")
    sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse = True)
    for x in sorted_pairs[:15]:
       print(x)
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
        print("start looking at variant")
        print(f"variant: {x}")
        pos_current = x[0]
        if pos_current in alignment_map.keys():
            pos_target = alignment_map[pos_current] # find equivalent position in target
            aa_target = target_seq_nospaces[pos_target]

        else:
            aa_target = ""
            pos_target = None
        current_seq_nospaces =  current_seq.replace(" ", "")
  
        
 
        print("curr no spaces", current_seq_nospaces)
        print("len no spaces", len(current_seq_nospaces))     
        print("pos current", pos_current)
      
        print("minus 1",current_seq_nospaces[pos_current - 1])
        #print("normal ", current_seq_nospaces[pos_current])     
        aa_current = current_seq_nospaces[pos_current]
         

        print(f"pos_current: {pos_current}, pos_target: {pos_target}, aa_current: {aa_current}, aa_target: {aa_target}")
        print(f"targ seq: {target_seq_nospaces}")
        print(f"orig seq: {orig_seq_nospaces}")
        print(f"prev seq: {current_seq_nospaces}")

        if aa_current != aa_target:

            # get score
            print(f"replacing {aa_current}-{pos_current} with {aa_target}")
            candidate_seq_nospaces = replace_char_at_index(current_seq_nospaces, pos_current, aa_target)
            print(f"new seq : {current_seq_nospaces}")

            candidate_seq = " ".join(candidate_seq_nospaces)
            candidate_ids = tokenizer(candidate_seq, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                candidate_representation = model.encoder(input_ids=candidate_ids)[0]
                candidate_mean_embedding = torch.mean(candidate_representation, dim=1)
        
            cossim_candidate_orig = F.cosine_similarity(candidate_mean_embedding, orig_mean_embedding) 
            cossim_candidate_target =  F.cosine_similarity(candidate_mean_embedding, target_mean_embedding) 

            # Not known yet if this is even necessary            
            if cossim_candidate_target > cossim_current_target_hist[-1]: 
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
                cossim_value = cossim_candidate_target.item()
                current_seqrecord =  SeqRecord(Seq(current_seq_nospaces), 
   
                       id=f"{orig_name}_{step}_{aa_current}{pos_current + 1}{aa_target}_{cossim_value:.3f}")
                output_seqrecords.append(current_seqrecord)
                output_scores.append(cossim_value)
                # The first two sequences are target and query
                alignment = do_align(alignment, [current_seqrecord]) # Add on the latest seqrecord
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
print("cossim_current_targ:", cossim_current_target)     
    
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
print(highest_score_seq.id)
print(highest_score_seq.seq)
p2_start = highest_score_seq

# Get all three aligned, orig, mod, target
#  starging alignment = target, query, so add on the highest scoring seq at the end
p2_alignment = do_align(alignment, [highest_score_seq])#[target_seqrecord]) #[target_seqrecord, orig_seqrecord, highest_score_seq])
print("p2 alignment", p2_alignment)



# keep the highest scoring sequence, and do again. 

# currently just showing that there are ones you can revert while keeping the high score


p2_output_list = [target_seqrecord]

mod_history = []
change_record = []


#starting alignment = target, orig, etc. 
orig_seq_aln = p2_alignment[1].seq
target_seq_aln = p2_alignment[0].seq
mod_seq_aln =  p2_alignment[2].seq

print("orig_seq_aln  ", orig_seq_aln)
print("mod_seq_aln   ", mod_seq_aln)
print("target_seq_aln", target_seq_aln)

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
          
          with torch.no_grad():
               candidate_ids = tokenizer(" ".join(p2_candidate.replace("-", "")), return_tensors="pt").input_ids.to(device)
               candidate_representation = model.encoder(input_ids=candidate_ids)[0]
               candidate_mean_embedding = torch.mean(candidate_representation, dim=1)
       
               cossim_candidate_orig = F.cosine_similarity(candidate_mean_embedding, orig_mean_embedding) 
               cossim_candidate_target =  F.cosine_similarity(candidate_mean_embedding, target_mean_embedding) 
               #print(pos, p2_alignment[1].seq[pos], p2_alignment[0].seq[pos])
               #print(cossim_candidate_orig.item(), cossim_candidate_target.item())
               modified_seqs.append((p2_candidate, cossim_candidate_target.item(), pos, mod_seq_aln[pos], orig_seq_aln[pos] ))

    if len(modified_seqs) > 0: 
        
        best_score = max(modified_seqs, key=lambda x: x[1])
        # pos, 
        change_record.append([best_score[2], best_score[3], best_score[4]])
        print("current best score", best_score[1])
        print("Best scoring seq", best_score[0])
        highest_score_seq = SeqRecord(Seq(best_score[0]), id=f"del{best_score[2] + 1}_{orig_name}_{best_score[1]:.3f}")
 
        mod_seq_aln = best_score[0]

        # Do we need to keep redoing this alignment?
        # Could it affect the position tracking?
        # Yes
        #p2_alignment = do_align([orig_seqrecord, highest_score_seq, target_seqrecord])
       

        p2_output_list.append(highest_score_seq)
    else:
        break

for x in change_record:
        print(x)


print("list", p2_output_list)
p2_output_list = p2_output_list + [orig_seqrecord]
print("appe", p2_output_list)
p2_output_alignment = do_align(alignment, p2_output_list)
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

    print("new mod", mod_seq_aln)
     #mod_seqrecord = 
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
        candidate_representation = model.encoder(input_ids=candidate_ids)[0]
        candidate_mean_embedding = torch.mean(candidate_representation, dim=1)
       
        cossim_candidate_orig = F.cosine_similarity(candidate_mean_embedding, orig_mean_embedding) 
        cossim_candidate_target =  F.cosine_similarity(candidate_mean_embedding, target_mean_embedding) 
    print(change, cossim_candidate_orig.item(), cossim_candidate_target.item()) 
    mod_seqrecord = SeqRecord(Seq(mod_seq_aln), id=f"{change[2]}_aln{change[0] + 1}-sq{true_pos_output}_{change[1]}_{orig_name}_{cossim_candidate_target.item():.3f}")
    p3_output_list.append(mod_seqrecord)

p3_output_list.append(target_seqrecord)

p3_output_alignment = do_align(alignment, p3_output_list)
print(p3_output_alignment)
with open(f"{outfile_base}_phase3_output.aln", "w") as handle:
     AlignIO.write(p3_output_alignment, handle, "clustal")

with open(f"{outfile_base}_phase3_output.aln.fasta", "w") as handle:
     AlignIO.write(p3_output_alignment, handle, "fasta")

with open(f"{outfile_base}_phase3_output.fasta", "w") as handle:
     SeqIO.write(p3_output_list, handle, "fasta")


