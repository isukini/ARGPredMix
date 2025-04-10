# utils.py
import faiss
import csv
import numpy as np
import os
import time
from feature import *
import torch
import numpy as np
import torch.nn as nn
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

# Define list of antibiotic resistance gene (ARG) types
ARGtype = [
    'beta-lactam',
    'peptide',
    'macrolide',
    'fluoroquinolone',
    'aminoglycoside',
    'tetracycline',
    'lincosamide',
    'streptogramin',
    'phenicol',
    'aminocoumarin',
    'disinfecting',
    'alkaloids',
    'diaminopyrimidine',
    'glycopeptide',
    'sulfonamide',
    'pleuromutilin',
    'nitroimidazole',
    'rifamycin',
    'sulfone',
    'oxazolidinone',
    'elfamycin',
    'other'
]

# Load mean and standard deviation values for normalization
def get_mean_std():
    with open('./model/mean_std1.pkl', 'rb') as f:
        data = pickle.load(f)
    mean1 = data['mean']
    std1 = data['std']
    with open('./model/features_stats2.pkl', 'rb') as f:
        stats = pickle.load(f)
    mean2 = stats['mean']
    std2 = stats['std']
    return mean1, std1, mean2, std2

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=300,
    max_depth=7,
    learning_rate=0.2,
    subsample=1.0
)

# FAISS KNN classifier using IVF index
class FaissIVFANNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=3, gpu=True, n_centroids=256):
        self.n_neighbors = n_neighbors
        self.n_centroids = n_centroids
        self.index = None
        self.classes_ = None
        self.train_labels = None
        self.gpu = gpu

    # Load training data and build index
    def load(self, data):
        X = data["features"]
        y = data["labels"]
        n_samples, n_features = X.shape

        X = np.ascontiguousarray(X.astype('float32'))

        # Create IVF index with L2 distance
        quantizer = faiss.IndexFlatL2(n_features)
        self.index = faiss.IndexIVFFlat(quantizer, n_features, self.n_centroids, faiss.METRIC_L2)

        # Train index if not already trained
        if not self.index.is_trained:
            self.index.train(X)

        self.index.add(X)

        # Transfer index to GPU if applicable
        if self.gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Save labels and class info
        self.train_labels = y
        self.classes_ = [np.unique(y[:, i]) for i in range(y.shape[1])]
        return self

    # Predict using FAISS index
    def predict(self, X):
        X = np.ascontiguousarray(X.astype('float32'))

        # Transfer index back to CPU for prediction
        if hasattr(self.index, 'is_on_gpu') and self.index.is_on_gpu:
            self.index = faiss.index_gpu_to_cpu(self.index)

        # Perform search
        D, I = self.index.search(X, self.n_neighbors)

        # Predict based on neighbor majority vote
        predictions = []
        for neighbors in I:
            neighbor_labels = self.train_labels[neighbors]
            pred = np.array([
                np.bincount(col, minlength=len(self.classes_[i])).argmax()
                for i, col in enumerate(neighbor_labels.T)
            ])
            predictions.append(pred)

        return np.array(predictions)

# Interleave and return ndarray and tensor as combined list
def save_interleaved_ndarray_tensor_to_csv(ndarray, tensor):
    list_a = ndarray.flatten().tolist()
    list_b = tensor.flatten().tolist()
    combined_data = []

    # Interleave values from two lists
    for i in range(len(list_a)):
        combined_data.append(list_a[i])
        combined_data.append(list_b[i])
    return combined_data

# CNN + BiLSTM Model 1
class CNN_BiLSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNN_BiLSTM1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32 * (input_size // 4), hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), 1, -1)
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# CNN + BiLSTM Model 2
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32 * (input_size // 4), hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), 1, -1)  # Flatten
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out = self.sigmoid(out)
        return out

# Write CSV header line
def write_header_to_csv(csvfile, nucleotide):
    csvwriter = csv.writer(csvfile)
    if nucleotide == 1:
        csvwriter.writerow([
            'Fasta File', 'Sequence Name','Gene Sequences', 'Amino Acid Sequences', 'resistant gene', 'score', 
            'Resistance category', 'ARG_identity', 'CARD_identity', 
            'beta-lactam', 'beta-lactam_score', 'peptide', 'peptide_score', 'macrolide', 
            'macrolide_score', 'fluoroquinolone', 'fluoroquinolone_score', 'aminoglycoside', 
            'aminoglycoside_score', 'tetracycline', 'tetracycline_score', 'lincosamide', 
            'lincosamide_score', 'streptogramin', 'streptogramin_score', 'phenicol', 
            'phenicol_score', 'aminocoumarin', 'aminocoumarin_score', 'disinfecting', 
            'disinfecting_score', 'alkaloids', 'alkaloids_score', 'diaminopyrimidine', 
            'diaminopyrimidine_score', 'glycopeptide', 'glycopeptide_score', 'sulfonamide', 
            'sulfonamide_score', 'pleuromutilin', 'pleuromutilin_score', 'nitroimidazole', 
            'nitroimidazole_score', 'rifamycin', 'rifamycin_score', 'sulfone', 'sulfone_score', 
            'oxazolidinone', 'oxazolidinone_score', 'elfamycin', 'elfamycin_score'
        ])
    else:
        csvwriter.writerow([
            'Fasta File', 'Sequence Name', 'Amino Acid Sequence', 'resistant gene', 'score', 
            'Resistance category', 'ARG_identity', 'CARD_identity', 
            'beta-lactam', 'beta-lactam_score', 'peptide', 'peptide_score', 'macrolide', 
            'macrolide_score', 'fluoroquinolone', 'fluoroquinolone_score', 'aminoglycoside', 
            'aminoglycoside_score', 'tetracycline', 'tetracycline_score', 'lincosamide', 
            'lincosamide_score', 'streptogramin', 'streptogramin_score', 'phenicol', 
            'phenicol_score', 'aminocoumarin', 'aminocoumarin_score', 'disinfecting', 
            'disinfecting_score', 'alkaloids', 'alkaloids_score', 'diaminopyrimidine', 
            'diaminopyrimidine_score', 'glycopeptide', 'glycopeptide_score', 'sulfonamide', 
            'sulfonamide_score', 'pleuromutilin', 'pleuromutilin_score', 'nitroimidazole', 
            'nitroimidazole_score', 'rifamycin', 'rifamycin_score', 'sulfone', 'sulfone_score', 
            'oxazolidinone', 'oxazolidinone_score', 'elfamycin', 'elfamycin_score'
        ])

# Write one row of prediction results to CSV
def write_to_csv(csvfile, fasta_file, seq_name, nucleotide_seq, amino_acid_seq, predictions, y_pred_probs, result, arg_identity, card_identity, combined_data, nucleotide=0):
    csvwriter = csv.writer(csvfile)
    y_pred_probs = round(y_pred_probs.item(), 4)
    if nucleotide == 1:
        row = [fasta_file, seq_name, nucleotide_seq, amino_acid_seq, predictions, y_pred_probs, result, arg_identity, card_identity]
    else:
        row = [fasta_file, seq_name, amino_acid_seq, predictions, y_pred_probs, result, arg_identity, card_identity]
    row.extend(combined_data[:])
    csvwriter.writerow(row)

# Process a single amino acid sequence through the full pipeline
def process_sequence(amino_acid_seq, esm_model, alphabet, device, mean1, std1, mean2, std2, resistseek1, resistclass2, xgb_classifier, knn, radical, threshold):
    # Truncate overly long sequences
    if len(amino_acid_seq) > 2000:
        amino_acid_seq = amino_acid_seq[:2000]

    # Generate ESM features
    esm_feature = esmfeature(amino_acid_seq, esm_model, alphabet, device)
    esm_feature1 = (esm_feature - mean1) / std1
    feature = torch.tensor(esm_feature1, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        feature = feature.reshape(1, 1, 2560)
        outputs1_cnn = resistseek1(feature)
        outputs1_cnn = outputs1_cnn.cpu().squeeze()
    outputs1_cnn = np.array(outputs1_cnn)

    y_pred_probs = 0
    if radical == 1:
        y_pred_probs = outputs1_cnn
    if radical == 0:
        xgb_feature = xgboostfeature(amino_acid_seq)
        xgb_feature = xgb_feature.reshape(1, -1)
        outputs1_xgb = xgb_classifier.predict_proba(xgb_feature)
        outputs1_xgb = outputs1_xgb[0][1]
        y_pred_probs = 0.39 * outputs1_xgb + 0.61 * outputs1_cnn
#         y_pred_probs = 2 * outputs1_xgb + 8 * outputs1_cnn

    predictions = (y_pred_probs > threshold).astype(int)
    result = ""
    predictions2 = None
    y_pred_probs_weighted2 = None

    if predictions == 1:
        # Use second classifier for detailed resistance type prediction
        esm_feature2 = (esm_feature - mean2) / std2
        esm_feature2 = torch.tensor(esm_feature2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = resistclass2(esm_feature2.to(device))
            outputs2_cnn = (torch.sigmoid(outputs).cpu().detach().numpy())

        if radical == 1:
            y_pred_probs_weighted2 = outputs2_cnn
        if radical == 0:
            knn_feature = knnfeature(amino_acid_seq)
            knn_feature = knn_feature.reshape(1, -1)
            outputs2_knn = knn.predict(knn_feature)
            outputs2_knn = np.array(outputs2_knn)
#             y_pred_probs_weighted2 = 0.34 * outputs2_knn + 0.66 * outputs2_cnn
            y_pred_probs_weighted2 = 0.2 * outputs2_knn + 0.8 * outputs2_cnn

        predictions2 = (y_pred_probs_weighted2 > 0.5).astype(int)
        y_pred_probs_weighted2 = np.round(y_pred_probs_weighted2, 4).astype(np.float64)

        selected_items = [ARGtype[i] for i in range(len(predictions2[0])) if predictions2[0, i] == 1]
        result = ','.join(selected_items)

    return predictions, y_pred_probs, result, predictions2, y_pred_probs_weighted2
