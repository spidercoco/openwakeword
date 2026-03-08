import os
import argparse
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
import matplotlib.pyplot as plt

def calculate_recall(y_true, y_pred):
    y_pred_tag = (y_pred > 0.5).float()
    correct_results_sum = (y_pred_tag * y_true).sum().float()
    total_positives = y_true.sum().float()
    recall = correct_results_sum / (total_positives + 1e-8)
    return recall.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_features", type=str, required=True)
    parser.add_argument("--negative_features", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # 输出目录
    output_dir = os.path.dirname(args.output_model)

    X_pos = np.load(args.positive_features, mmap_mode='r')
    X_neg = np.load(args.negative_features, mmap_mode='r')

    y_pos = np.ones(X_pos.shape[0])
    y_neg = np.zeros(X_neg.shape[0])

    X = np.concatenate([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 96, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 记录训练历史
    history = {'loss': [], 'recall': []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            all_preds.append(outputs.detach())
            all_targets.append(batch_y.detach())
        
        # 计算该 epoch 的平均指标
        epoch_loss = total_loss / len(loader)
        epoch_recall = calculate_recall(torch.cat(all_targets), torch.cat(all_preds))
        
        history['loss'].append(epoch_loss)
        history['recall'].append(epoch_recall)
        
        print(f"PROGRESS:{epoch + 1}/{args.epochs}")
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Recall: {epoch_recall:.4f}", flush=True)

    # 导出模型
    model.eval()
    dummy_input = torch.zeros((1, 28, 96))
    torch.onnx.export(model, dummy_input, args.output_model)

    # 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label="loss")
    plt.plot(history['recall'], label="recall")
    plt.legend()
    plt.ylim(0, 1.1)
    plt.title("Training Loss and Recall")
    plt.savefig(os.path.join(output_dir, "training_plot.png"))
    plt.close()

    # 保存最终指标
    metrics = {
        "final_recall": history['recall'][-1],
        "final_loss": history['loss'][-1]
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()
