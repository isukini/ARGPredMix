import torch
import pickle
import numpy as np
import torch.nn as nn
import esm
import numpy as np
import pandas as pd
import re

def clean_sequence(sequence):
    """Replace any character that is not one of the 20 common amino acids with 'X'."""
    sequence = re.sub(r"\*", "", sequence)  # Remove stop codon '*'
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", sequence)

def load_esm_model(device, model_path='./model/esm2_t36_3B_UR50D.pt'):
    """Pre-load the ESM model."""
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    model.eval()
    model = model.to(device)
    return model, alphabet, device

def generate_esm_features(sequence, model, alphabet, device):
    """Encode a single amino acid sequence."""
    sequence = clean_sequence(sequence)
    batch_converter = alphabet.get_batch_converter()
    data_batch = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data_batch)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36])
    token_embeddings = results["representations"][36]

    sequence_length = (batch_tokens[0] != alphabet.padding_idx).sum()
    sequence_embedding = token_embeddings[0, 1:sequence_length - 1].mean(0).cpu().numpy()

    return sequence_embedding
