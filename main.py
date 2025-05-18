import torch
from model import FasterSNN
from util import create_dataloaders, evaluate_all_metrics
from test import load_model

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    train_loader, test_loader = create_dataloaders(
        train_dir='/home/wuchangwei/dataset/AIBLnew/train',
        test_dir='/home/wuchangwei/dataset/AIBLnew/test',
        batch_size=16
    )

    # 构建模型
    model = FasterSNN(num_classes=3, time_steps=2).to(device)
    model_name = "FasterSNN"

    # 加载模型权重
    model = load_model(model_name, model, device)

    # 全面评估
    metrics = evaluate_all_metrics(model, test_loader, model_name, device)

if __name__ == '__main__':
    main()