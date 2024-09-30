from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from config import device, ml_model_function, project_root, xgboost_params
import xgboost as xgb
import numpy as np
import joblib
import torch
import os

seq_len = 10


def train_xgboost():
    model_dir = os.path.join(project_root, 'data/model')
    data_path = os.path.join(project_root, 'data/encode/edge_encode.npy')
    data = np.load(data_path)
    # 1. 按滑动窗口提取数据
    data = np.lib.stride_tricks.sliding_window_view(data, window_shape=seq_len + 1, writeable=False)
    # 2. 构造输入及目标序列
    x = torch.tensor(data[:, :-1], device=device).cpu().numpy()
    y = torch.tensor(data[:, -1], dtype=torch.long, device=device).cpu().numpy()
    # 将数据转换为DMatrix格式
    dtrain = xgb.DMatrix(x, label=y)
    # 定义模型参数
    params = xgboost_params
    # 训练模型
    num_round = 6  # 迭代次数
    bst = xgb.train(params, dtrain, num_round)
    # 在测试集上进行预测
    preds = bst.predict(dtrain)
    # 计算准确率
    accuracy = accuracy_score(y, preds)
    print("Accuracy:", accuracy)
    # 保存模型
    model_path = "xgboost.json"
    bst.save_model(os.path.join(model_dir, model_path))


def train_svm():
    model_dir = os.path.join(project_root, 'data/model')
    data_path = os.path.join(project_root, 'data/encode/edge_encode.npy')
    data = np.load(data_path)
    # 1. 按滑动窗口提取数据
    data = np.lib.stride_tricks.sliding_window_view(data, window_shape=seq_len + 1, writeable=False)
    # 2. 构造输入及目标序列
    x = torch.tensor(data[:, :-1], device=device).cpu().numpy()
    y = torch.tensor(data[:, -1], dtype=torch.long, device=device).cpu().numpy()
    svclassifier = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr', random_state=42)
    svclassifier.fit(x, y)
    preds = svclassifier.predict(x)
    accuracy = accuracy_score(y, preds)
    print("Accuracy:", accuracy)
    model_path = "svclassifier.joblib"
    joblib.dump(svclassifier, os.path.join(model_dir, model_path))


def train_decision_tree():
    model_dir = os.path.join(project_root, 'data/model')
    data_path = os.path.join(project_root, 'data/encode/edge_encode.npy')
    data = np.load(data_path)
    
    # 1. 按滑动窗口提取数据
    data = np.lib.stride_tricks.sliding_window_view(data, window_shape=seq_len + 1, writeable=False)
    
    # 2. 构造输入及目标序列
    x = torch.tensor(data[:, :-1], device=device).cpu().numpy()
    y = torch.tensor(data[:, -1], dtype=torch.long, device=device).cpu().numpy()
 
    # 创建并训练决策树模型
    model = DecisionTreeClassifier()
    model.fit(x, y)
    
    # 在测试集上进行预测
    preds = model.predict(x)
    accuracy = accuracy_score(y, preds)
    print("Accuracy:", accuracy)
    
    # 保存模型
    model_path = "decision_tree.joblib"
    joblib.dump(model, os.path.join(model_dir, model_path))


def train_naive_bayes():
    model_dir = os.path.join(project_root, 'data/model')
    data_path = os.path.join(project_root, 'data/encode/edge_encode.npy')
    data = np.load(data_path)
    # 1. 按滑动窗口提取数据
    data = np.lib.stride_tricks.sliding_window_view(data, window_shape=seq_len + 1, writeable=False)
    # 2. 构造输入及目标序列
    x = torch.tensor(data[:, :-1], device=device).cpu().numpy()
    y = torch.tensor(data[:, -1], dtype=torch.long, device=device).cpu().numpy()
    # 这里直接使用MultinomialNB
    model = MultinomialNB()
    model.fit(x,y)
    preds = model.predict(x)
    accuracy = accuracy_score(y, preds)
    print("Accuracy:", accuracy)
    model_path = "naive_bayes.joblib"
    joblib.dump(model, os.path.join(model_dir, model_path))



def train_ml_model(method_name: str):
    if method_name in ml_model_function:
        eval(ml_model_function[method_name])()  # 调用对应的函数
    else:
        print("Function not found for ml method:", method_name)
