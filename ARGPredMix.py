import os
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
from load_model import all_model
from feature import *
from utils import ARGtype, write_to_csv, save_interleaved_ndarray_tensor_to_csv, process_sequence, get_mean_std, write_header_to_csv
import argparse  # For command-line argument parsing
import subprocess
# Parse command-line arguments
def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="ARGPredMix - Predict resistance genes and output results to CSV")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input FASTA file path")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output CSV file path")
    parser.add_argument('-blast', '--blast', action='store_true', help="Enable BLAST for antibiotic resistance gene alignment")
    parser.add_argument('-n', '--nucleotide', action='store_true', help="Input is nucleotide sequence or whole genome sequence")
    parser.add_argument('-all', '--all', action='store_true', help="Enable writing all prediction results to CSV")
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help="Set the prediction threshold, default: 0.8")
    parser.add_argument('-r', '--radical', action='store_true', help="Enable radical mode")
    parser.add_argument('-clean', '--clean', action='store_true', help="Clean up temporary files")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Validate input arguments
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} does not exist!")
        return

    # Get input and output file paths
    fasta_file = args.input
    csv_file = args.output
    threshold = args.threshold  # Get the threshold parameter

    # Set blast and writeall options based on command-line arguments
    blast = 1 if args.blast else 0
    wirteall = 1 if args.all else 0
    radical = 1 if args.radical else 0
    nucleotide = 1 if args.nucleotide else 0
    clean = 1 if args.clean else 0
    # Print parameters for confirmation
    print(f"Input FASTA file: {fasta_file}")
    print(f"Output CSV file: {csv_file}")
    print(f"Enable BLAST: {blast}")
    print(f"Enable writing all results: {wirteall}")
    print(f"Enable radical mode: {wirteall}")
    print(f"Prediction threshold set to: {threshold}")
    
    # Load models and related data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        resistseek1, resistclass2, xgb_classifier, knn, model, alphabet = all_model(device)
        mean1, std1, mean2, std2 = get_mean_std()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    records = None
    fnn_records = None
    if nucleotide == 1:
        if not os.path.exists('tmp'):
            try:
                subprocess.run(f'mkdir tmp', shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to execute 'mkdir tmp': {e}")
        try:
            with open(f'./tmp/{fasta_file}prodigal_output.log', 'w') as log_file:
                subprocess.run(f'prodigal -i {fasta_file} -a tmp/tmp.faa -d tmp/tmp.fnn -p meta', shell=True, check=True, stdout=log_file, stderr=log_file)
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute 'prodigal  -i {fasta_file} -a tmp/tmp.faa -d tmp/tmp.fnn': {e}")
        records = list(SeqIO.parse('tmp/tmp.faa', "fasta"))
        fnn_records = list(SeqIO.parse('tmp/tmp.fnn', "fasta"))
    else:
        # Read FASTA file
        records = list(SeqIO.parse(fasta_file, "fasta"))

    if fnn_records is None:
        fnn_records = [None] * len(records)  # Create a list of None values
    num_sequences = len(records)
    print(f"There are {num_sequences} sequences in the FASTA file.")
    
    # Open CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        write_header_to_csv(csvfile, nucleotide)
        for record, fnn_record in tqdm(zip(records, fnn_records), desc="Processing sequences", total=len(records)):
            seq_name = record.id
            amino_acid_seq = str(record.seq)
            nucleotide_seq = None
            if fnn_record:
                nucleotide_seq = str(fnn_record.seq)
            predictions, y_pred_probs, result, predictions2, y_pred_probs_weighted2 = process_sequence(
                amino_acid_seq, model, alphabet, device, mean1, std1, mean2, std2, 
                resistseek1, resistclass2, xgb_classifier, knn, radical, threshold)  # Pass threshold parameter
            arg_identity = None
            card_identity = None
            # Write prediction results to CSV
            if predictions == 0 and wirteall == 1:
                write_to_csv(csvfile, fasta_file, seq_name, nucleotide_seq, amino_acid_seq, predictions, y_pred_probs, result="", arg_identity=arg_identity, card_identity=card_identity, combined_data=[], nucleotide=nucleotide)
            if predictions == 1:
                if blast == 1:
                    arg_identity = getidentity(amino_acid_seq, 'arg')
                    card_identity = getidentity(amino_acid_seq, 'card')
                write_to_csv(csvfile, fasta_file, seq_name, nucleotide_seq, amino_acid_seq, predictions, y_pred_probs, result, arg_identity=arg_identity, card_identity=card_identity, combined_data=save_interleaved_ndarray_tensor_to_csv(predictions2, y_pred_probs_weighted2), nucleotide=nucleotide)
    
    if clean == 1 and nucleotide == 1:
        subprocess.run(f'rm -rf tmp', shell=True, check=True)

if __name__ == "__main__":
    main()

