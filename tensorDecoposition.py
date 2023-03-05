import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import sys

class TensorDecoposition(nn.Module):
    def __init__(self, args):
        super(TensorDecoposition, self).__init__()
        self.gpu = args.ifgpu
        self.hd = args.targetHiddenDim  # 250
        self.rel_hd = args.relationHiddenDim  # 70/8
        self.num_heads = args.relationNum  # 24/211
        self.qk_hd=args.relationLinearpara # 100



        # 这里需要初始化两个实体句子特征的线性层(N*100)，一个关系特征矩阵(relationNum*relationNum/3)，一个core tensor(100*100*80)

        self.q_linear = nn.Linear(self.hd, self.qk_hd, bias=True)
        self.k_linear = nn.Linear(self.hd, self.qk_hd, bias=True)
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0)
        nn.init.constant_(self.k_linear.bias, 0)

        self.core_tensor=nn.Parameter(torch.Tensor(self.qk_hd,self.qk_hd,self.rel_hd))
        self.core_tensor=init.xavier_uniform_(self.core_tensor)

        self.rel_emb=nn.Parameter(torch.Tensor(self.num_heads,self.rel_hd))
        self.rel_emb=init.xavier_uniform_(self.rel_emb)
        self.l1_loss = 0





    def forward(self, relationDecoderResult_1,relationDecoderResult_2):

    

        query = relationDecoderResult_1.contiguous()  # batch_size*seq_len*hd
        key = relationDecoderResult_2.contiguous()
        q = self.q_linear(query)
        k = self.k_linear(key) 

        '''
        size q : batch_size*seq*qk_hd
        size k : batch_size*seq*qk_hd
        size rel_emb : num_r*rel_hd
        size core_tensor : qk_hd*qk_hd*rel_hd
        '''

        reconstruct_tensor=torch.einsum('ijk,knd->ijnd',[q,self.core_tensor])
        reconstruct_tensor=torch.einsum('imn,ijnd->ijmd',[k,reconstruct_tensor])
        reconstruct_tensor=torch.einsum('rd,ijmd->ijmr',[self.rel_emb,reconstruct_tensor])
        reconstruct_tensor=torch.einsum('ijmr->irjm',[reconstruct_tensor])



        return reconstruct_tensor

