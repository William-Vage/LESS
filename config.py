from pipeline.edge.model import *
import torch
import os


project_root = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cpu')

model_name_appendex = {
    'lstm': 'lstm.pt',
    'gru': 'gru.pt',
    'bigru': 'bigru.pt',
    'attentiongru': 'attentiongru.pt',
    '1dcnn': '1dcnn.pt',
    'xgboost': 'xgboost.json',
    'svm': 'svm.joblib',
    'naive_bayes': 'naive_bayes.joblib',
    'decision_tree': 'decision_tree.joblib'
}

ml_model_list = ['xgboost', 'decision_tree', 'naive_bayes']  # 'svm',
deep_model_list = ['lstm','gru','1dcnn', 'bigru']

ml_model_function = {
    'xgboost': 'train_xgboost',
    'svm': 'train_svm',
    'naive_bayes': 'train_naive_bayes',
    'decision_tree': 'train_decision_tree'
}

deep_model_info = {
    'lstm': (LeonardLstm, 'lstm.pt'),
    'gru': (MyGRU, 'gru.pt'),
    'bigru': (MyBiGRU, 'bigru.pt'),
    'attentiongru': (AttentionGRUModel, 'attentiongru.pt'),
    '1dcnn': (CNNClassifier, '1dcnn.pt')
}

xgboost_params = {
    'objective': 'multi:softmax',  # The loss function is multi-class softmax
    'num_class': 13,  # Number of categories is 0 to 12, a total of 13 categories
    'eval_metric': 'merror',  # Evaluation metric is multi-class error rate
    'eta': 0.1,  # Learning rate
    'max_depth': 5,  # Maximum depth of a tree
    'subsample': 0.8,  # Proportion of samples used for training each tree
    'colsample_bytree': 0.8,  # Proportion of features used by each tree
    'seed': 42  # Random seed
}