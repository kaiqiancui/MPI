# 在 policy.py 文件中，FCNetwork 下方添加这个新类
import torch.nn as nn
class AttentionHead(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, hidden_dim=256):
        super().__init__()
        # PyTorch 的 MultiheadAttention 层
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=nhead, batch_first=True)
        
        # 注意力层之后接一个 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm 通常和 Attention 配合得很好
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # MultiheadAttention 需要的输入形状是 (Batch, Sequence, Features)
        # 我们的特征 x 的形状是 (Batch, Features)，所以需要增加一个序列维度
        seq_in = x.unsqueeze(1) # 形状变为 (Batch, 1, Features)
        
        # Query, Key, Value 都是它自己，进行自注意力计算
        attn_output, _ = self.attention(seq_in, seq_in, seq_in)
        
        # 移除序列维度，形状恢复为 (Batch, Features)
        attn_output = attn_output.squeeze(1)
        
        # 通过最后的 MLP 得到动作
        action = self.mlp(attn_output)
        return action