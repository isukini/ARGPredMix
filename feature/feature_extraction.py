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
bmpso= [0, 1, 2, 3, 6, 8, 10, 11, 12, 13, 15, 17, 19, 20, 22, 24, 26, 27, 29, 31, 32, 34, 37, 38, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51, 54, 56, 57, 59, 62, 63, 64, 65, 70, 71, 73, 76, 77, 78, 80, 82, 85, 86, 87, 89, 90, 91, 93, 94, 95, 96, 97, 98, 100, 104, 105, 106, 107, 108, 110, 111, 115, 116, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 131, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 146, 149, 151, 153, 154, 155, 157, 158, 161, 162, 163, 164, 165, 166, 168, 171, 172, 173, 175, 176, 178, 182, 184, 185, 186, 188, 190, 192, 193, 194, 196, 197, 201, 203, 204, 208, 209, 210, 213, 215, 216, 217, 221, 222, 223, 227, 228, 229, 230, 231, 232, 237, 240, 241, 242, 243, 248, 249, 250, 253, 255, 258, 259, 261, 262, 263, 264, 266, 268, 269, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 284, 285, 287, 288, 293, 295, 296, 297, 300, 301, 304, 305, 311, 312, 314, 315, 318, 321, 323, 326, 328, 329, 331, 333, 334, 335, 336, 340, 341, 342, 343, 345, 347, 348, 349, 350, 351, 352, 353, 354, 355, 361, 362, 363, 365, 366, 367, 368, 371, 373, 374, 375, 376, 377, 378, 380, 387, 388, 389, 392, 393, 394, 395, 396, 397, 398, 400, 402, 403, 404, 405, 407, 409, 411, 412, 413, 415, 417, 420, 421, 422, 423, 424, 426, 429, 430, 431, 432, 433, 434, 435, 439, 443, 448, 449, 451, 453, 454, 455, 456, 458, 459, 461, 463, 464, 465, 466, 467, 468, 470, 471, 472, 475, 476, 478, 479, 480, 490, 491, 493, 494, 500, 501, 502, 503, 504, 507, 508, 510, 512, 514, 517, 519, 521, 522, 524, 526, 527, 528, 530, 532, 535, 537, 539, 547, 548, 549, 551, 552, 557, 560, 565, 568, 570, 571, 575, 576, 577, 580, 583, 584, 585, 586, 587, 588, 589, 590, 593, 595, 597, 598, 600, 601, 602, 603, 605, 606, 608, 610, 611, 613, 614, 616, 617, 618, 619, 620, 622, 630, 631, 632, 633, 634, 635, 637, 639, 642, 643, 644, 646, 647, 649, 651, 652, 653, 655, 656, 658, 659, 660, 662, 663, 664, 666, 668, 671, 673, 675, 676, 678, 679, 680, 682, 683, 684, 685, 688, 690, 692, 693, 698, 700, 702, 706, 707, 708, 709, 710, 711, 712, 713, 715, 716, 717, 719, 720, 721, 722, 724, 726, 727, 728, 730, 731, 732, 733, 735, 736, 737, 739, 740, 741, 742, 746, 748, 750, 751, 754, 755, 757, 760, 762, 765, 768, 770, 772, 773, 774, 777, 778, 779, 780, 781, 783, 785, 787, 789, 790, 791, 792, 794, 795, 796, 797, 801, 806, 807, 809, 810, 811, 815, 816, 820, 821, 822, 823, 824, 825, 826, 828, 829, 831, 833, 834, 836, 837, 838, 839, 841, 842, 843, 844, 848, 852, 854, 856, 857, 858, 859, 860, 862, 863, 864, 865, 866, 868, 871, 872, 873, 875, 876, 877, 883, 889, 890, 895, 897, 899, 901, 902, 903, 904, 905, 906, 909, 910, 911, 912, 913, 918, 919, 920, 921, 923, 925, 926, 927, 929, 930, 931, 932, 936, 937, 939, 942, 943, 944, 945, 946, 947, 948, 949, 952, 953, 955, 956, 961, 962, 965, 967, 970, 971, 974, 975, 976, 977, 978, 979, 980, 983, 984, 985, 986, 988, 989, 994, 997, 999, 1000, 1001, 1003, 1006, 1009, 1012, 1014, 1015, 1017, 1019, 1021, 1023, 1026, 1028, 1034, 1035, 1036, 1037, 1038, 1040, 1041, 1042, 1045, 1046, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1057, 1058, 1060, 1062, 1063, 1066, 1068, 1069, 1072, 1075, 1077, 1078, 1079, 1081, 1082, 1083, 1085, 1087, 1090, 1092, 1095, 1096, 1100, 1106, 1107, 1109, 1110, 1115, 1116, 1120, 1123, 1124, 1125, 1126, 1128, 1129, 1138, 1141, 1143, 1144, 1145, 1146, 1150, 1152, 1154, 1156, 1158, 1160, 1165, 1166, 1170, 1171, 1172, 1173, 1174, 1178, 1181, 1185, 1186, 1187, 1188, 1189, 1190, 1192, 1194, 1195, 1196, 1197, 1199, 1205, 1207, 1208, 1209, 1212, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224]
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
    features = features[bmpso]
    # print(features.shape)
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

    


def esmfeature(sequence, model, alphabet, device):
    """Generate ESM features for a given sequence using the pre-trained ESM model."""
    # Generate features for the given sequence using ESM model
    return generate_esm_features(sequence, model, alphabet, device)


def getidentity(sequence, db):
    """Compute the sequence identity by comparing with a database."""
    # Get the maximum sequence identity by comparing with the provided database
    return get_max_identity(sequence, db)

    
