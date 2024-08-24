from joblib import load
import torch
from cv2 import imread
import numpy as np
import os

from model_files.CNN import CNN
from model_files.feature_vector_generation import get_patch_yi

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_feature_vector(image_path: str, model):
    feature_vector = np.empty((1, 400))
    feature_vector[0, :] = get_patch_yi(model, imread(image_path))
    return feature_vector

def load_pretained_models():
    # Load the pretrained CNN with the CASIA2 dataset
    with torch.no_grad():
        cnn_model = CNN()
        cnn_model_path = os.path.join(ROOT_DIR, "pre_trained_cnn", "CASIA2_WithRot_LR001_b128_nodrop.pt")
        cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=lambda storage, loc: storage))
        cnn_model.eval()
        cnn_model = cnn_model.double()

    # Load the pretrained svm model
    svm_model_path = os.path.join(ROOT_DIR, "pre_trained_svm", "CASIA2_WithRot_LR001_b128_nodrop.pt")
    svm_model = load(svm_model_path)
    return [cnn_model, svm_model]


def classify(input_image_path):
    [cnn_model, svm_model] = load_pretained_models()
    image_feature_vector = get_feature_vector(input_image_path, cnn_model)
    return svm_model.predict(image_feature_vector)