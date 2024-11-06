import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T

from tvm import te, tir
from tvm.relax.frontend import nn as tvm_nn
from tvm.relax.frontend.nn import Tensor, op

class CustomLSTM(tvm_nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,  
                 batch_first=False, dropout=0., bidirectional=False):  
        super().__init__()  
        self.input_size = input_size  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.bias = bias  
        self.batch_first = batch_first  
        self.dropout = dropout  
        self.bidirectional = bidirectional  
        self.num_directions = 2 if bidirectional else 1  

        # 为每一层和每一个方向创建权重与偏置  
        self.weight_ih = []  
        self.weight_hh = [] 
        self.bias_ih = []  
        self.bias_hh = []

        for layer in range(num_layers):  
            for direction in range(self.num_directions):  
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions  
                # 输入到隐藏层的权重
                idx = layer * self.num_directions + direction
                setattr(  
                    self,   
                    f"w_ih_{idx}",   
                    tvm_nn.Parameter(shape=(4 * hidden_size, layer_input_size), dtype="float32")  
                )
                 
                setattr(  
                    self,   
                    f"w_hh_{idx}",   
                    tvm_nn.Parameter(shape=(4 * hidden_size, hidden_size), dtype="float32")  
                )  
                if bias:  
                    setattr(  
                        self,   
                        f"bias_ih_{idx}",   
                        tvm_nn.Parameter(shape=(4 * hidden_size,), dtype="float32")  
                    )  
                    setattr(  
                        self,   
                        f"bias_hh_{idx}",   
                        tvm_nn.Parameter(shape=(4 * hidden_size,), dtype="float32")  
                    )
 
        self.dropout_layer = None   

    def forward(self, input):  
        # 处理 batch_first  
        if self.batch_first:  
            input = op.permute_dims(input, [1, 0, 2])  # 转换为 (seq_len, batch, input_size)  

        seq_len, batch_size, _ = input.shape

        h0 = op.zeros((self.num_layers * self.num_directions,  
                             batch_size, self.hidden_size))  
                            
        c0 = op.zeros((self.num_layers * self.num_directions,  
                             batch_size, self.hidden_size))   

        # 分离初始隐藏状态和细胞状态  
        h0 = op.reshape(h0, (self.num_layers, self.num_directions, batch_size, self.hidden_size))  
        c0 = op.reshape(c0, (self.num_layers, self.num_directions, batch_size, self.hidden_size))  

        # 初始化输出  
        output = input  
        h_n = []  
        c_n = []  

        # 遍历每一层  
        for layer in range(self.num_layers):  
            layer_output = []  
            layer_h_n = []  
            layer_c_n = []  
            layer_expr = Tensor.from_scalar(layer,"int32")

            for direction in range(self.num_directions):  
                # 获取对应层和方向的权重
                dir_expr = Tensor.from_scalar(direction,"int32")
                idx = layer * self.num_directions + direction
                # w_ih = self.weight_ih[idx]  
                # w_hh = self.weight_hh[idx]  
                # if self.bias:  
                #     b_ih = self.bias_ih[idx]  
                #     b_hh = self.bias_hh[idx]  
                # else:  
                #     b_ih = b_hh = None  

                # 选择时间步遍历方向  
                if direction == 0:  # 前向  
                    time_range = range(seq_len)  
                else:  # 反向  
                    time_range = range(seq_len - 1, -1, -1)  

                # 初始化隐藏状态和细胞状态  
                h = op.take(op.take(h0, layer_expr, axis=0), dir_expr, axis=0)
                c = op.take(op.take(c0, layer_expr, axis=0), dir_expr, axis=0)  

                # 存储每个时间步的输出  
                dir_output = []  
                for t in time_range:
                    t_expr = Tensor.from_scalar(t,"int32")
                    x = op.take(output, t_expr, axis=0)  # 修正此处，直接使用 t  
                    gates = op.matmul(x, op.permute_dims( getattr(self, f"w_ih_{idx}") )) + op.matmul(h, op.permute_dims(getattr(self, f"w_hh_{idx}") ))  
                    if self.bias:  
                        gates += getattr(self, f"bias_ih_{idx}") + getattr(self, f"bias_hh_{idx}")  
                    # 切分各个门  
                    i_gate, f_gate, g_gate, o_gate = op.chunk(gates, 4, 1)  
                    i_gate = op.sigmoid(i_gate)  
                    f_gate = op.sigmoid(f_gate)  
                    g_gate = op.tanh(g_gate)  
                    o_gate = op.sigmoid(o_gate)  
                    # 计算新的细胞状态和隐藏状态  
                    c = f_gate * c + i_gate * g_gate  
                    h = o_gate * op.tanh(c)  
                    dir_output.append(h)  

                # 如果是反向，反转输出以恢复原始时间顺序  
                if direction == 1:  
                    dir_output = dir_output[::-1]  
                unsqueezed_tensors = [op.unsqueeze(t, 0) for t in dir_output]  

                # 使用 torch.cat 在新维度上连接所有张量  
                dir_output = op.concat(unsqueezed_tensors, dim=0) 
                # dir_output = torch.stack(dir_output, dim=0)  # (seq_len, batch, hidden_size) stack
                layer_output.append(dir_output)  
                layer_h_n.append(h)  
                layer_c_n.append(c)  

            # 合并双向输出  
            if self.bidirectional:  
                output = op.concat(layer_output, dim=2)  # (seq_len, batch, hidden_size * 2)  
            else:  
                output = layer_output[0]  

            # 保存最终的隐藏状态和细胞状态  
            for h, c in zip(layer_h_n, layer_c_n):  
                h_n.append(h)  
                c_n.append(c)  

            # 应用 dropout，除了最后一层  
            if self.dropout_layer is not None and layer < self.num_layers - 1:  
                output = self.dropout_layer(output)  
        print(output.shape)
        # 组织最终的隐藏状态和细胞状态
        unsqueezed_h_n = [op.unsqueeze(t, 0) for t in h_n]  
        h_n = op.concat(unsqueezed_h_n, dim=0)
        unsqueezed_c_n = [op.unsqueeze(t, 0) for t in c_n]  
        c_n = op.concat(unsqueezed_c_n, dim=0) 
        # h_n = torch.stack(h_n, dim=0)  # (num_layers * num_directions, batch, hidden_size)  
        # c_n = torch.stack(c_n, dim=0)  # (num_layers * num_directions, batch, hidden_size)  

        # 处理 batch_first  
        if self.batch_first:
            output = op.permute_dims(output, [1, 0, 2])  # 转换为 (batch, seq_len, hidden_size * num_directions)
            # output = output.transpose(0, 1)  # 转换回 (batch, seq_len, hidden_size * num_directions)  

        return output
