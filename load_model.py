import torch
import esm
from utils import ARGtype, write_to_csv, FaissIVFANNClassifier, save_interleaved_ndarray_tensor_to_csv, CNN_BiLSTM1, CNN_BiLSTM, xgb_classifier, get_mean_std
from xgboost import XGBClassifier
import pickle
from utils import FaissIVFANNClassifier
from feature import loadesm

def load_resistseek_model(device):
    """Load the ResistSeek CNN BiLSTM model"""
    model = CNN_BiLSTM1(2560, 128, 2, 1).to(device)
    model.load_state_dict(torch.load('./model/resistseek_cnnbilstm.pth'))
    model.eval()
    print("ResistSeek CNN BiLSTM Model loaded from file.")
    return model

def load_resistclass2_model(device):
    """Load the ResistClass2 CNN BiLSTM model"""
    model = CNN_BiLSTM(2560, 128, 2, 22).to(device)
    model.load_state_dict(torch.load('./model/resistclass_cnnbilstm.pth'))
    model.eval()
    print("ResistClass CNN BiLSTM Model loaded from file.")
    return model

def load_xgb_classifier():
    """Load the XGBoost classifier model"""
    model = XGBClassifier()
    model.load_model("./model/resistseek_xgb.json")
    print("ResistSeek XGB Classifier Model loaded from file.")
    return model

def load_knn_classifier():
    """Load the Faiss KNN classifier model"""
    with open("./model/resistclass_mlknn_data.pkl", "rb") as file:
        loaded_data = pickle.load(file)  
    knn = FaissIVFANNClassifier(n_neighbors=3, gpu=False, n_centroids=1)
    knn.load(loaded_data)
    print("ResistClass KNN Model loaded from file.")
    return knn

def load_esm_model(device):
    """Load the ESM model"""
    model, alphabet = loadesm(device)
    print("ESM2 model loaded from file.")
    return model, alphabet

def load_mean_std():
    """Load the standardized mean and standard deviation values"""
    mean1, std1, mean2, std2 = get_mean_std()
    return mean1, std1, mean2, std2

def all_model(device):
    """Load all models (ResistSeek, ResistClass2, XGBoost, KNN, ESM)"""
    resistseek1 = load_resistseek_model(device)
    resistclass2 = load_resistclass2_model(device)
    xgb_classifier = load_xgb_classifier()
    knn = load_knn_classifier()
    model, alphabet = load_esm_model(device)
    return resistseek1, resistclass2, xgb_classifier, knn, model, alphabet
