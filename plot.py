import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

def generate_classification_report(true_labels, pred_labels, model_name, target_names=None):
    from sklearn.metrics import classification_report
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