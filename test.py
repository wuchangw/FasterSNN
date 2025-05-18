import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import os

def evaluate_model(model, test_loader, device, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def load_model(model_name, model, device):
    weight_path = f'./FasterSNN.pth'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件 {weight_path} 不存在！")

    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        state_dict = {k: v for k, v in checkpoint.items()
                     if not any(s in k for s in ['total_ops', 'total_params'])}
        model.load_state_dict(state_dict, strict=False)
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                     if not any(s in k for s in ['total_ops', 'total_params'])}
        model.load_state_dict(state_dict, strict=False)
    else:
        try:
            model.load_state_dict(checkpoint, strict=False)
        except:
            raise ValueError("无法识别的权重文件格式")

    print(f"成功加载权重: {weight_path}")
    return model
