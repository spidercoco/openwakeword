import torch
from torch import optim, nn
import torchinfo
import torchmetrics
import copy
import os
import sys
import numpy as np
import collections
import argparse
import logging
import yaml
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from openwakeword.data import mmap_batch_generator

# 设置日志
logging.basicConfig(level=logging.INFO)

# ================= 严格同步 oww_train.py 的 Model 类 =================
class Model(nn.Module):
    def __init__(self, n_classes=1, input_shape=(16, 96), model_type="dnn",
                 layer_dim=128, n_blocks=1, seconds_per_example=None):
        super().__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        
        # 严格匹配 FCN 结构
        class FCNBlock(nn.Module):
            def __init__(self, layer_dim):
                super().__init__()
                self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                self.relu = nn.ReLU()
                self.layer_norm = nn.LayerNorm(layer_dim)
            def forward(self, x):
                return self.relu(self.layer_norm(self.fcn_layer(x)))

        class Net(nn.Module):
            def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=1):
                super().__init__()
                self.flatten = nn.Flatten()
                self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
                self.relu1 = nn.ReLU()
                self.layernorm1 = nn.LayerNorm(layer_dim)
                self.blocks = nn.ModuleList([FCNBlock(layer_dim) for i in range(n_blocks)])
                self.last_layer = nn.Linear(layer_dim, n_classes)
                self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()
            def forward(self, x):
                x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
                for block in self.blocks: x = block(x)
                return self.last_act(self.last_layer(x))
        
        self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes)
        self.recall = torchmetrics.Recall(task='binary')
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.history = collections.defaultdict(list)
        self.loss = torch.nn.functional.binary_cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def lr_warmup_cosine_decay(self, global_step, warmup_steps, hold, total_steps, target_lr=1e-3):
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
        warmup_lr = target_lr * (global_step / warmup_steps)
        if hold > 0: learning_rate = np.where(global_step > warmup_steps + hold, learning_rate, target_lr)
        return np.where(global_step < warmup_steps, warmup_lr, learning_rate)

    def auto_train(self, X_train, X_val, steps, max_negative_weight, target_id):
        # 简化版自动训练，保持计算逻辑
        self.to(self.device)
        self.model.to(self.device)
        
        weights = np.linspace(1, max_negative_weight, int(steps)).tolist()
        val_steps = np.linspace(steps-int(steps*0.25), steps, 10).astype(np.int64)
        
        logging.info(f"Starting training for {steps} steps...")
        
        for step_ndx, data in enumerate(X_train):
            if step_ndx >= steps: break
            
            x, y = data[0].to(self.device), data[1].to(self.device)
            y_ = y[..., None].to(torch.float32)

            lr = self.lr_warmup_cosine_decay(step_ndx, warmup_steps=steps//5, hold=steps//3, total_steps=steps, target_lr=0.0001)
            for g in self.optimizer.param_groups: g['lr'] = lr

            self.optimizer.zero_grad()
            predictions = self.model(x)
            
            # 权重逻辑
            w = torch.ones(y.shape[0]) * weights[step_ndx]
            w[y == 1] = 1
            
            loss = self.loss(predictions, y_, w.to(self.device)[..., None])
            loss.backward()
            self.optimizer.step()
            
            # 进度输出
            if (step_ndx + 1) % 10 == 0 or (step_ndx + 1) == steps:
                print(f"PROGRESS:{step_ndx+1}/{int(steps)}", flush=True)
                
            if step_ndx in val_steps:
                # 记录指标用于图表 (逻辑同步 train_web.py)
                self.history['loss'].append(loss.item())
                self.best_val_recall = self.recall(predictions, y_).item()
                self.history['recall'].append(self.best_val_recall)

        return self.model

    def export_model(self, model, model_name, output_dir):
        model_to_save = copy.deepcopy(model)
        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],
                          os.path.join(output_dir, model_name + ".onnx"), opset_version=13)

# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    task_root = os.path.dirname(os.path.abspath(args.config))
    
    # 路径解析
    pos_train_feat = os.path.join(task_root, "positive_features_train.npy")
    neg_train_feat = os.path.join(task_root, "negative_features_train.npy")
    pos_test_feat = os.path.join(task_root, "positive_features_test.npy")
    neg_test_feat = os.path.join(task_root, "negative_features_test.npy")

    # 获取输入形状
    test_data = np.load(pos_test_feat, mmap_mode='r')
    input_shape = test_data.shape[1:]

    oww = Model(n_classes=1, input_shape=input_shape, model_type=config.get("model_type", "dnn"),
                layer_dim=config.get("layer_size", 128))

    # 数据转换逻辑
    def transform(x, n=input_shape[0]):
        if n != x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
        return x

    feature_files = config.get("feature_data_files", {}).copy()
    # 补充本项目生成的特征
    feature_files['positive'] = pos_train_feat
    feature_files['adversarial_negative'] = neg_train_feat
    
    # 处理路径
    base_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(task_root))))
    for k, v in feature_files.items():
        if k not in ['positive', 'adversarial_negative']:
            feature_files[k] = os.path.join(base_project_root, v.replace("./", ""))

    label_transforms = {k: (lambda x: [1 for i in x] if k == 'positive' else lambda x: [0 for i in x]) for k in feature_files.keys()}
    data_transforms = {k: transform for k in feature_files.keys()}

    batch_generator = mmap_batch_generator(
        feature_files,
        n_per_class=config.get("batch_n_per_class", {"positive": 32, "adversarial_negative": 32}),
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms
    )

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator): self.generator = generator
        def __iter__(self): return self.generator

    X_train = torch.utils.data.DataLoader(IterDataset(batch_generator), batch_size=None)

    # 运行训练
    best_model = oww.auto_train(
        X_train=X_train, X_val=None, # 简化验证
        steps=config["steps"],
        max_negative_weight=config.get("max_negative_weight", 1000),
        target_id=task_root
    )

    # 导出
    oww.export_model(model=best_model, model_name="beary_custom", output_dir=task_root)

    # 保存图表与指标
    plt.figure()
    plt.plot(oww.history['loss'], label="loss")
    plt.plot(oww.history['recall'], label="recall")
    plt.legend(); plt.ylim(0, 1.1)
    plt.savefig(os.path.join(task_root, "training_plot.png"))
    
    with open(os.path.join(task_root, "metrics.json"), "w") as f:
        json.dump({"final_recall": oww.best_val_recall}, f)

if __name__ == "__main__":
    main()
