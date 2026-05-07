# 核心训练循环 (精简伪代码)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for x, y in dataloader: # DataLoader 自动打包连续内存
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)             # 前向推理
        loss = criterion(pred, y)   # 计算 MSE 误差
        loss.backward()             # 反向传播
        optimizer.step()            # 权重更新



# 修复 1: 压实显存碎片
def forward(self, x):
    self.lstm.flatten_parameters() # 终极防御：强制权重在显存中连续排列
    out, _ = self.lstm(x)
    return self.fc(out)

# 修复 2: 绕过 cuDNN 长序列封锁
with torch.no_grad():
    with torch.backends.cudnn.flags(enabled=False): # 强制使用原生 CUDA 计算
        y_pred_tensor = model(x_tensor)



# ==========================================
# model_anomaly_detect.py 核心侦测逻辑
# ==========================================
# 1. AI 理论基准 (健康预估)
with torch.no_grad():
    y_pred = model(x_tensor).squeeze().cpu().numpy() * 4.0

# 2. 传感器实际回传数据
y_actual = df['thrust'].values

# 3. 计算绝对残差并触发报警
residual = np.abs(y_actual - y_pred)
THRESHOLD = 0.6  # 工业安全阈值 0.6 N

# 判定异常区间，用于前端标红
is_anomaly = residual > THRESHOLD


# ==========================================
# model_finetune.py 迁移学习核心逻辑
# ==========================================
# 1. 挂载已具备通用物理规律的基准模型
model = SimpleLSTM().to(device)
model.load_state_dict(torch.load("thruster_lstm_v1.pth"))

# 2. 冻结策略与极小学习率
# 采用 1e-4 的学习率，防止遗忘之前的通用规律
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. 仅使用 SN13 的专属地面测试数据进行微调
for epoch in range(20): # 仅跑少量 Epoch
    # ... 计算 loss 并反向传播 ...
    
torch.save(model.state_dict(), "thruster_lstm_finetuned_sn13.pth")



# ==========================================
# model_shap_analysis.py 核心排雷代码
# ==========================================
# 强制 CPU 模式，规避 GPU 下 LSTM 的底层解释 Bug
device = torch.device('cpu') 

# 编写定制化的 Wrapper，将时序维度压平
class ShapWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        out = self.base_model(x)
        return out.mean(dim=1) # 降维：输出整个序列的平均推力

# 注入 SHAP 深度解释引擎
explainer = shap.DeepExplainer(ShapWrapper(base_model), background_tensor)
shap_values = explainer.shap_values(test_tensor)