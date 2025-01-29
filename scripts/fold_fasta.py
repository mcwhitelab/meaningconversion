import os
import subprocess
import time
from pathlib import Path
from Bio import SeqIO
import re

def fold_sequences_with_esmfold(fasta_path):
    """
    Fold all sequences in a FASTA file using ESMFold API.
    Creates a directory named after the FASTA file to store PDB outputs.
    """
    # Create output directory next to the fasta file
    fasta_path = Path(fasta_path)
    fasta_name = fasta_path.stem
    output_dir = fasta_path.parent / f"esmfold_pdbs_{fasta_name}"
    output_dir.mkdir(exist_ok=True)
    
    # First pass to determine max step number for padding
    max_step = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        step_match = re.search(r'step(\d+)', record.id)
        if step_match:
            max_step = max(max_step, int(step_match.group(1)))
    
    # Calculate padding width based on max step
    pad_width = len(str(max_step))
    
    # Read sequences from FASTA
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Extract and pad step number
        step_match = re.search(r'step(\d+)', record.id)
        if step_match:
            step_num = step_match.group(1)
            padded_step = str(int(step_num)).zfill(pad_width)
            # Replace the original step number with padded version
            pdb_name = record.id.replace(f"step{step_num}", f"step{padded_step}")
        else:
            pdb_name = record.id
            
        pdb_path = output_dir / f"{pdb_name}.pdb"
        
        # Skip if PDB already exists
        if pdb_path.exists():
            print(f"Skipping {pdb_name} - already exists")
            continue
        
        # Prepare sequence (remove any whitespace)
        sequence = str(record.seq).replace(" ", "")
        
        # Construct and execute curl command
        cmd = [
            "curl",
            "-X", "POST",
            "--data", sequence,
            "https://api.esmatlas.com/foldSequence/v1/pdb/"
        ]
        
        print(f"Folding sequence for {pdb_name}...")
        try:
            # Run curl command and capture output
            result = subprocess.run(cmd, 
                                 capture_output=True, 
                                 text=True, 
                                 check=True)
            
            # Save PDB file
            with open(pdb_path, 'w') as f:
                f.write(result.stdout)
            
            print(f"Successfully saved {pdb_name}.pdb")
            
            # Sleep to avoid overwhelming the API
            time.sleep(2)
            
        except subprocess.CalledProcessError as e:
            print(f"Error folding {pdb_name}: {e}")
            print(f"Error output: {e.stderr}")
            continue
        except Exception as e:
            print(f"Unexpected error for {pdb_name}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fold sequences using ESMFold API')
    parser.add_argument('fasta_file', 
                       type=str,
                       help='Path to input FASTA file')
    
    args = parser.parse_args()
    
    fold_sequences_with_esmfold(args.fasta_file)
