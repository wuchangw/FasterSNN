import torch
import torch.nn as nn
from model import FasterSNN
import torch.optim as optim
from tqdm import tqdm
from util import create_dataloaders, evaluate_all_metrics
import time
import argparse


def train_model(model, train_loader, val_loader, num_epochs=20, lr=1e-3, weight_decay=1e-3, device='cuda'):
    criterion = nn.CrossEntropyLoss()

    # Adam with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f" - LR scheduler: ReduceLROnPlateau(factor=0.5, patience=3)")
    print(f" - Training epochs: {num_epochs}")

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        epoch_time = time.time() - start_time

        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"\nEpoch {epoch + 1}/{num_epochs} - "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'FasterSNN.pth')
            print(f"*** New best model saved with accuracy: {best_acc:.2f}% ***")

    print(f"\nTraining completed. Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    return model


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100. * correct / total


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 regularization)')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Example usage - you'll need to define your model and data loaders
    print("Training configuration:")
    print(f" - Epochs: {args.epochs}")
    print(f" - Learning rate: {args.lr}")
    print(f" - Weight decay: {args.weight_decay}")
    print(f" - Device: {args.device}")

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = FasterSNN(num_classes=3, time_steps=2).to(device)
    train_loader, test_loader = create_dataloaders(
        train_dir='/home/wuchangwei/dataset/AIBLnew/train',
        test_dir='/home/wuchangwei/dataset/AIBLnew/test',
        batch_size=16
    )
    train_model(model, train_loader, test_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=args.device)

if __name__ == '__main__':
    main()
