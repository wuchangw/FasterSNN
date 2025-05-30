import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score

def evaluate_all_metrics(model, test_loader, model_name, device):
    from test import evaluate_model
    from plot import (generate_classification_report, 
                     plot_confusion_matrix, 
                     plot_roc_curve)
    
    true_labels, pred_labels, pred_probs = evaluate_model(model, test_loader, device)
    
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'kappa': cohen_kappa_score(true_labels, pred_labels),
    }

    generate_classification_report(true_labels, pred_labels, model_name,
                                 target_names=["AD", "MCI", "CN"])
    plot_confusion_matrix(true_labels, pred_labels, model_name)
    metrics['avg_auc'] = plot_roc_curve(true_labels, pred_probs, model_name)

    print("\n=== Model Performance Summary ===")
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")
    print(f"Avg AUC: {metrics['avg_auc']:.4f}")
    return metrics

def create_dataloaders(train_dir, test_dir, batch_size=16):
    from dataloader import ADNIDataset
    
    train_data = ADNIDataset(root_dir=train_dir, target_size=(64, 64, 64))
    test_data = ADNIDataset(root_dir=test_dir, target_size=(64, 64, 64))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
