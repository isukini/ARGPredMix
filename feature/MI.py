import math
from collections import Counter
from itertools import product
import numpy as np
import math
from collections import Counter
import re
import numpy as np

def clean_sequence(sequence):
    """Replace any character that is not one of the 20 common amino acids with 'X'."""
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", sequence)

def calculate_mutual_information_for_all_pairs(sequence, amino_acids):
    # Count the frequency of each amino acid (marginal probability)
    counts_x = Counter(sequence)
    total_count = len(sequence)
    probabilities_x = {x: counts_x.get(x, 0) / total_count for x in amino_acids}

    # Count the joint probability (p(x, y)) for all possible pairs, initialize as 0
    pair_counts = {pair: 0 for pair in product(amino_acids, repeat=2)}

    # Count the occurrences of actual amino acid pairs (joint probability)
    pairs = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
    observed_pair_counts = Counter(pairs)

    # Fill in the actual pair frequencies
    total_pairs = len(pairs)
    for pair, count in observed_pair_counts.items():
        pair_counts[pair] = count / total_pairs

    # Calculate mutual information for each pair
    mi_per_pair = {}
    total_mutual_information = 0

    for (x, y), p_xy in pair_counts.items():
        p_x = probabilities_x.get(x, 0)
        p_y = probabilities_x.get(y, 0)

        if p_x > 0 and p_y > 0 and p_xy > 0:
            # Calculate mutual information for this pair
            mi = p_xy * math.log2(p_xy / (p_x * p_y))
            # Correct any negative values caused by floating point precision errors
            if mi < 0:
                mi = 0
            mi_per_pair[(x, y)] = mi
            total_mutual_information += mi
        else:
            # For unseen pairs, mutual information is 0
            mi_per_pair[(x, y)] = 0

    return mi_per_pair, total_mutual_information

def Mutual_information(sequence):
    # Calculate mutual information
    amino_acids = list("ACDEFGHIKLMNPQRSTVWYX")
    mi_per_pair, total_mutual_information = calculate_mutual_information_for_all_pairs(sequence, amino_acids)

    # Extract mutual information for each amino acid pair as features
    mi_values = [mi_per_pair[pair] for pair in product(amino_acids, repeat=2)]

    # Add total mutual information as the last feature
    mi_values.append(total_mutual_information)

    return np.array(mi_values)

# Define a function to calculate conditional mutual information
def calculate_conditional_mutual_information(sequence, amino_acids):
    triplet_counts = Counter()  # Count the occurrences of all triplets (xzy)
    pair_xz_counts = Counter()  # Count the occurrences of pairs (xz)
    pair_yz_counts = Counter()  # Count the occurrences of pairs (yz)
    z_counts = Counter()  # Count the occurrences of the middle amino acid (z)

    for i in range(len(sequence) - 2):
        x, z, y = sequence[i], sequence[i + 1], sequence[i + 2]
        triplet_counts[(x, z, y)] += 1
        pair_xz_counts[(x, z)] += 1
        pair_yz_counts[(y, z)] += 1
        z_counts[z] += 1

    total_triplets = sum(triplet_counts.values())
    conditional_mi = {}
    total_mi = 0

    for x in amino_acids:
        for z in amino_acids:
            for y in amino_acids:
                p_xyz = triplet_counts[(x, z, y)] / total_triplets if (x, z, y) in triplet_counts else 0
                p_xz = pair_xz_counts[(x, z)] / total_triplets if (x, z) in pair_xz_counts else 0
                p_yz = pair_yz_counts[(y, z)] / total_triplets if (y, z) in pair_yz_counts else 0
                p_z = z_counts[z] / total_triplets if z in z_counts else 0

                if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                    # Calculate conditional mutual information
                    mi = p_xyz * math.log2(p_xyz / (p_xz * p_yz / p_z))
                    conditional_mi[(x, z, y)] = mi
                    total_mi += mi
                else:
                    conditional_mi[(x, z, y)] = 0

    return conditional_mi, total_mi


def Conditional_mutual_information(sequence):
    # Define the amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWYX")

    # Clean the sequence to handle non-standard amino acids
    sequence = clean_sequence(sequence)

    # Calculate the conditional mutual information features
    conditional_mi, total_mi = calculate_conditional_mutual_information(sequence, amino_acids)

    # Convert the conditional mutual information for each triplet and the total mutual information to a feature vector
    feature_vector = list(conditional_mi.values()) + [total_mi]

    return np.array(feature_vector)

# Define a function to compute the power spectrum
def power_spectrum(seq):
    # Compute the Fourier transform of the sequence
    fourier_transform = np.fft.fft(seq)
    # Calculate the power spectrum
    power_spectrum = np.abs(fourier_transform) ** 2
    # Remove the first frequency component (DC component)
    power_spectrum = power_spectrum[1:]
    return power_spectrum

# Define a function for the cumulative power spectrum
def cumulative_power_spectrum(power_spectrum):
    # Compute the cumulative sum of the power spectrum
    cps = np.cumsum(power_spectrum)
    return cps

# Calculate the moment vector M_j
def moment_vector(cps, N_base, N_total, j):
    # Factor used in moment calculation
    factor = 1 / ((N_base * (N_total - N_base)) ** (j - 1) * N_total ** j)
    # Calculate the moment of the cumulative power spectrum
    moment = factor * np.sum(cps ** j)
    return moment

# Calculate the central moment vector CM_j
def central_moment_vector(cps, N_base, N_total, j, mean_cps):
    # Factor used in central moment calculation
    factor = 1 / ((N_base * (N_total - N_base)) ** (j - 1) * N_total ** j)
    # Calculate the central moment of the cumulative power spectrum
    central_moment = factor * np.sum(np.abs(cps - mean_cps) ** j)
    return central_moment

# Extract features from the sequence
def extract_features(sequence, j):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    N_total = len(sequence)  # Total length of the sequence

    M_vector = []
    CM_vector = []
    for aa in amino_acids:
        # Encode the current amino acid as 1, others as 0
        binary_sequence = [1 if residue == aa else 0 for residue in sequence]

        # Compute the power spectrum and cumulative power spectrum
        ps = power_spectrum(binary_sequence)
        cps = cumulative_power_spectrum(ps)

        # Get the count of the current amino acid in the sequence
        N_base = sequence.count(aa)

        if N_base == 0:
            M_vector.append(0)
            CM_vector.append(0)
            continue

        # Calculate the mean of the cumulative power spectrum
        mean_cps = np.mean(cps) / (N_total - 1)

        # Calculate the moment and central moment vectors
        moment = moment_vector(cps, N_base, N_total, j)
        central_moment = central_moment_vector(cps, N_base, N_total, j, mean_cps)

        # Append the results to the corresponding vectors
        M_vector.append(moment)
        CM_vector.append(central_moment)

    return np.array(M_vector), np.array(CM_vector)

# Process a single amino acid sequence and return features
def Fourier_transform(sequence, j=2):
    cleaned_seq = clean_sequence(sequence)
    # Extract features with j = 2 as an example
    M_vector, CM_vector = extract_features(cleaned_seq, j)
    return M_vector, CM_vector
