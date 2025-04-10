import pandas as pd
import pickle
import re
import numpy as np
# from protlearn.features import cksaap
from itertools import product
from collections import Counter


def replace_non_standard_aa(sequences):
    """Replace non-standard amino acids with 'X'."""
    non_standard_aa_pattern = re.compile(r'[^ARNDCEQGHILKMFPSTWYV]')
    def replace_seq(seq):
        return non_standard_aa_pattern.sub('X', seq)
    return [replace_seq(seq) for seq in sequences]

def remove_x_from_sequences(sequences):
    """Remove 'X' characters from sequences."""
    return [seq.replace('X', '') for seq in sequences]

def extract_aaindex1_features(sequence, start=1, end=None):
    """Extract AAIndex1 features for the given sequence."""
    # Load AAIndex1 data (assumed to be a CSV file)
    aaind1 = pd.read_csv('/root/autodl-tmp/project/feature/data/aaindex1.csv')
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    desc = aaind1['Description'].values
    aaind1 = {aa: aaind1[aa].to_numpy() for aa in amino_acids}

    # Initialize the feature array
    features = np.zeros(553)  # AAIndex1 has 553 indices

    # Slice the sequence
    sequence = sequence[start - 1:end]

    # Initialize temporary array to hold the AAIndex1 values for the sequence
    tmp_arr = np.zeros((553, len(sequence)))

    # Extract features for the sequence
    for j, aa in enumerate(sequence):
        if aa in aaind1:  # Check if amino acid is valid
            tmp_arr[:, j] = aaind1[aa]

    # Compute the mean of the indices for the sequence
    features = tmp_arr.mean(axis=1)

    return features, desc



def cksaap(sequence, k=1, remove_zero_cols=False, start=1, end=None):
    """Extract CKSAAP features for the given sequence."""
    # Define amino acid types and dipeptide combinations
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    doublets = [''.join(p) for p in product(amino_acids, repeat=2)]
    patterns = [a[0] + '.' * k + a[1] for a in doublets]

    # Initialize feature vector
    features = np.zeros(len(patterns), dtype=int)

    # Slice sequence and compute features
    sequence = sequence[start - 1:end]
    for j, pattern in enumerate(patterns):
        features[j] = len(re.findall(r'(?=(' + pattern + '))', sequence))

    # Remove zero columns (if needed)
    if remove_zero_cols:
        non_zero_cols = features > 0
        features = features[non_zero_cols]
        patterns = [p for p, nz in zip(patterns, non_zero_cols) if nz]

    return features, patterns




def ctd(sequence, start=1, end=None):
    """Extract CTD features for the given sequence."""
    # Define amino acid categories
    classes = {
        'A': 1, 'G': 1, 'V': 1,
        'I': 2, 'L': 2, 'F': 2, 'P': 2,
        'Y': 3, 'M': 3, 'T': 3, 'S': 3,
        'H': 4, 'N': 4, 'Q': 4, 'W': 4,
        'R': 5, 'K': 5,
        'D': 6, 'E': 6,
        'C': 7
    }

    # Generate all possible triplet combinations
    ctd_list = [''.join(i) for i in product('1234567', repeat=3)]
    ctd_counts = {triad: 0 for triad in ctd_list}

    # Slice the sequence and map to categories
    sequence = sequence[start - 1:end]
    mapped_seq = ''.join(str(classes[aa]) for aa in sequence if aa in classes)

    # Calculate the number of triads
    triads = [mapped_seq[i:i + 3] for i in range(len(mapped_seq) - 2)]
    counts = Counter(triads)

    # Fill the feature vector
    for triad in ctd_list:
        ctd_counts[triad] = counts.get(triad, 0)

    # Return feature vector
    features = np.array(list(ctd_counts.values()), dtype=float)

    return features, ctd_list



def extract_dipeptide_features(sequence):
    """Extract dipeptide composition features for the given sequence."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    dipeptides = [a + b for a in amino_acids for b in amino_acids]

    # Extract all dipeptides from the sequence
    dipeptide_list = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    count = Counter(dipeptide_list)
    total = len(dipeptide_list)

    # Compute frequencies and construct feature vector
    feature_vector = [count[dp] / total if total > 0 else 0 for dp in dipeptides]
    return np.array(feature_vector)


def AAC(sequence):
    """Extract amino acid composition features for the given sequence."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWYX"
    count = Counter(sequence)  # Count the frequency of each amino acid
    total = len(sequence)  # Total length of the sequence
    # Compute the composition frequency for each amino acid
    aac_vector = [count[aa] / total for aa in amino_acids]
    return np.array(aac_vector)
