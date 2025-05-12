import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert (d_out % num_heads == 0),\
            "d_out must be divisible by num_heads" 
        # 这里是因为我们现在设置的d_out就是最终想要得到的context_vectors的embedding_size，因为多头注意力最终是将每个自注意力头的结果进行拼接，
        # 因此就需要将d_out的嵌入维度平均划分到每一个自注意头上进行处理，最终再拼接回来。

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #将投影的维度减少，以得到最终期望的d_out大小的嵌入维度

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias) #注意这里的输出维度是d_out，目的是仅做一次矩阵相乘得到一个大的映射之后的projection size，然后再根据num_heads的数量，将projection size平均划分到每个注意力头上；这样做的效率就比上一种堆叠+串行执行的方式效率高，因为那种方式是有多少个注意力头，就要分别初始化多少次映射矩阵，并且进行矩阵相乘。
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out) 
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # ======================================================================
        # Step 1: 将输入 X 线性映射为 Q，K，V
        queries = self.W_query(x) # shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        values = self.W_value(x)

        #通过增加'num_heads'维度，将Q,K,V矩阵划分到每个自注意力头上，具体实现：
        # 将最后一个维度从d_out划分为(num_heads, head_dim):
        # (b, num_tokens, d_out) ——> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose维度变换，为了便于矩阵乘法的进行，也就是模仿之前只有一个注意力头的时候的情况，让最后两个维度参与矩阵乘法
        # (n, num_tokens, num_heads, head_dim) ——>  (n, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # ======================================================================


        # ======================================================================
        # Step 2: 计算Scaled dot-product attention (启用mask)
        #计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 使用因果注意力mask进行掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] 
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重（归一化）
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)
        
        # Dropout
        attn_weights = self.dropout(attn_weights)

        # 计算最终的context vectors
        # 变换shape，为接下来concat做准备
        # (b, num_heads, num_tokens, head_dim) ——> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # ======================================================================

        # ======================================================================
        # Step 3: Concat每个注意力头的输出
        # concat所有注意力头的输出，得到最终的d_out embedding size
        # (b, num_tokens, num_heads, head_dim) ——> (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # ======================================================================
        
        # ======================================================================
        # Step 4: 将concat之后的结果进行线性映射
        #线性映射（可选），放在这里主要是因为常见的LLM都有这么一层，研究表明去掉之后也不会太影响模型的性能
        context_vec = self.out_proj(context_vec)
        # ======================================================================

        return context_vec