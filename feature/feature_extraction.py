from .MI import Fourier_transform
from .MI import Mutual_information
from .MI import Conditional_mutual_information
from .Sequence_features import extract_dipeptide_features
import numpy as np
from .Sequence_features import ctd
from .Sequence_features import cksaap
from .Sequence_features import extract_aaindex1_features
from .Sequence_features import AAC
from .esmfeature import generate_esm_features
from .esmfeature import load_esm_model
from .blastfeature import get_max_identity
import torch
# sequence = "ACDEFGHIKLMNPQRSTVWY" 

# amino_acids = list("ACDEFGHIKLMNPQRSTVWYX")

# features = Mutual_information(sequence)

# print(features.shape)

# features = Conditional_mutual_information(sequence)

# print(features.shape)

# features,features2 = Fourier_transform(sequence, j=2)

# print(features.shape)
# print(features2.shape)

# dipeptide_features = extract_dipeptide_features(sequence)
# print(dipeptide_features.shape)

# ctdd,_ = ctd(sequence)
# print(ctdd.shape)

# ctdd,_ = cksaap(sequence)
# print(ctdd.shape)

# ctdd,_ = extract_aaindex1_features(sequence)
# print(ctdd.shape)

# ctdd = AAC(sequence)
# print(ctdd.shape)

def xgboostfeature(sequence):
    """Generate feature vector for XGBoost model using various feature extraction methods."""
    # Compute mutual information for the sequence
    mi = Mutual_information(sequence)
    
    # Compute conditional mutual information for the sequence
    cmi = Conditional_mutual_information(sequence)
    
    # Extract dipeptide composition features
    dipeptide_features = extract_dipeptide_features(sequence)
    
    # Extract CKSAAP features for the sequence
    cksaap_f, _ = cksaap(sequence)
    
    # Compute Fourier transform features (j=1 and j=2)
    M_vector1, CM_vector1 = Fourier_transform(sequence, j=1)
    M_vector2, CM_vector2 = Fourier_transform(sequence, j=2)
    
    # Concatenate all the extracted features into a single feature vector
    features = np.concatenate([mi, cmi, dipeptide_features, cksaap_f, M_vector1, CM_vector1, M_vector2, CM_vector2])
    
    # Return the concatenated feature vector
    return features

def knnfeature(sequence):
    """Generate feature vector for KNN model using mutual information, CTD, and CKSAAP features."""
    # Compute mutual information for the sequence
    mi = Mutual_information(sequence)
    
    # Compute CTD features for the sequence
    ctd_f, _ = ctd(sequence)
    
    # Extract CKSAAP features for the sequence
    cksaap_f, _ = cksaap(sequence)
    
    # Concatenate mutual information, CTD features, and CKSAAP features into a single vector
    features = np.concatenate([mi, ctd_f, cksaap_f])
    
    # Return the concatenated feature vector
    return features


def loadesm(device):
    """Load the pre-trained ESM model."""
    # Load ESM model and alphabet
    model, alphabet, device = load_esm_model(device)
    
    # Return the loaded model, alphabet, and device
    return model, alphabet


def esmfeature(sequence, model, alphabet, device):
    """Generate ESM features for a given sequence using the pre-trained ESM model."""
    # Generate features for the given sequence using ESM model
    return generate_esm_features(sequence, model, alphabet, device)


def getidentity(sequence, db):
    """Compute the sequence identity by comparing with a database."""
    # Get the maximum sequence identity by comparing with the provided database
    return get_max_identity(sequence, db)

    