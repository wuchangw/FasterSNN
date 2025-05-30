import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import os
import argparse
from model import FasterSNN
from util import create_dataloaders


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


def generate_classification_report(true_labels, pred_labels, model_name, target_names=None):
    report = classification_report(true_labels, pred_labels, target_names=target_names, digits=4)
    print(f"Classification Report for {model_name}:\n{report}\n")

    os.makedirs('./report', exist_ok=True)
    report_path = f'./report/{model_name}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report for {model_name} saved to {report_path}")


def plot_confusion_matrix(true_labels, pred_labels, model_name):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["AD", "MCI", "CN"],
                yticklabels=["AD", "MCI", "CN"])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    os.makedirs('./picture/cm', exist_ok=True)
    cm_path = f'./picture/cm/{model_name}_confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix for {model_name} saved to {cm_path}")


def plot_roc_curve(true_labels, pred_probs, model_name, num_classes=3):
    lb = LabelBinarizer()
    lb.fit([0, 1, 2])
    true_labels_bin = lb.transform(true_labels)

    plt.figure(figsize=(8, 6))
    class_auc = []

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        class_auc.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    mean_auc = np.mean(class_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve (Avg AUC = {mean_auc:.2f})')
    plt.legend(loc="lower right")

    os.makedirs('./picture/roc', exist_ok=True)
    roc_path = f'./picture/roc/{model_name}_roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve for {model_name} saved to {roc_path}")

    return mean_auc


def main():
    parser = argparse.ArgumentParser(description='Test FasterSNN model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    _, test_loader = create_dataloaders(
        train_dir='/home/wuchangwei/dataset/AIBLnew/train',
        test_dir= '/home/wuchangwei/dataset/AIBLnew/test',
        batch_size=args.batch_size
    )

    # 加载模型
    model = FasterSNN(num_classes=3, time_steps=2).to(device)
    model_name = "FasterSNN"
    model = load_model(model_name, model, device)

    # 评估模型
    true_labels, pred_labels, pred_probs = evaluate_model(model, test_loader, device)

    # 生成报告和图表
    generate_classification_report(true_labels, pred_labels, model_name, target_names=["AD", "MCI", "CN"])
    plot_confusion_matrix(true_labels, pred_labels, model_name)
    avg_auc = plot_roc_curve(true_labels, pred_probs, model_name)

    # 计算并打印关键指标
    accuracy = accuracy_score(true_labels, pred_labels)
    kappa = cohen_kappa_score(true_labels, pred_labels)

    print("\n=== Model Performance Summary ===")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"Avg AUC: {avg_auc:.4f}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc

    main()
