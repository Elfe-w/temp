
from dgl.nn.pytorch import GraphConv
import numpy as np
import torch
import torch.nn as nn

class Self_Attention(nn.Module):
    def __init__(self,in_feats,d_k,d_v,device):
        super(Self_Attention,self).__init__()
        self.W_Q = GraphConv(in_feats, d_k)
        self.W_K = GraphConv(in_feats, d_k)
        self.W_V = GraphConv(in_feats, d_v)
        self.W_O = GraphConv(d_v,in_feats)
        self.d_k=d_k
        self.device=device
    def forward(self,g,inputs,h_attn=None):
        Q = self.W_Q(g, inputs)
        K = self.W_K(g, inputs)
        V = self.W_V(g, inputs)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.FloatTensor([self.d_k])).to(self.device)
        if h_attn == None:
            attn = nn.Softmax(dim=-1)(scores)
        else:
            print('self attention scores:',scores)
            attn = nn.Softmax(dim=-1)(scores+h_attn)
        attn_out = torch.matmul(attn, V)
        attn_out=self.W_O(g,attn_out)
        return attn_out, attn
        pass

class GCNFeedforwardLayer(nn.Module):
    def __init__(self, in_feats, hidden_size,dropout):
        super(GCNFeedforwardLayer, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, in_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        out = self.dropout(torch.relu(self.conv1(g,inputs)))
        out=self.conv2(g,out)
        return out

class HCLLayer(nn.Module):
    def __init__(self,in_feats,d_k,d_v,hideen_size,dropout,device):
        super(HCLLayer,self).__init__()
        self.in_feats=in_feats
        self.self_attention=Self_Attention(in_feats,d_k,d_v,device)
        self.ln=nn.LayerNorm(in_feats)
        self.feedforward=GCNFeedforwardLayer(in_feats,hideen_size,dropout)
        self.dropout=nn.Dropout(dropout)
    def forward(self,g,inputs,attn=None):
        attn_out,attn=self.self_attention(g,inputs,attn)
        attn_out=self.ln(attn_out)
        out=self.feedforward(g,attn_out+inputs)
        out=self.ln(out)
        return out,attn

'''
n_path_node:每条path有几个node（映射到几个node里面）
'''
class HCL(nn.Module):
    def __init__(self,n_path_node,n_layers,in_feats,d_k,d_v,hidden_size,dropout,num_class,device):
        super(HCL,self).__init__()
        self.device=device
        self.layers=nn.ModuleList([HCLLayer(in_feats,d_k,d_v,hidden_size,dropout,device) for _ in range(n_layers)])
        self.cla1 = nn.Linear(in_feats,128)
        self.cla2 = nn.Linear(128, num_class)
        self.n_path_node=n_path_node
        self.path = nn.Linear(self.n_path_node*in_feats,in_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self,g,node_emb,path_emb,path_node_dict=None,attn=None):
        path_emb=path_emb.repeat(self.n_path_node,1)
        for path in path_node_dict:
            for node in path_node_dict[path]:
                node_emb[node] = node_emb[node] + path_emb[path]
        for layer in self.layers:
            node_emb,attn =layer(g,node_emb,attn)
        fe = self.dropout(torch.relu(self.cla1(node_emb)))
        out=self.cla2(fe)
        return out,fe
'''
直接拼接加池化求和
'''
def fefusion_pooling(node_num,path_num,node_emb,path_emb):
    import copy
    cat_polling=False
    gating=False
    if len(node_num)!=len(path_num):
        print('特征數量不匹配，無法進行特征融合')
        return []
    else:
        node_begin,path_begin=0,0
        ast=[]
        for i in range(len(node_num)):
            node_slice=copy.deepcopy(node_emb[node_begin:node_begin+node_num[i]])
            path_slice=copy.deepcopy(path_emb[path_begin:path_begin+path_num[i]])
            node_begin=node_begin+node_num[i]
            path_begin=path_begin+path_num[i]
            ast_temp=torch.cat((node_slice,path_slice),0)
            ast.append(torch.sum(ast_temp,0).numpy())
        tensor_data=torch.Tensor(ast)
        return tensor_data


def pooling(node_num,node_emb):
    node_begin = 0
    node,path=[],[]
    for i in range(len(node_num)):
        node_slice = node_emb[node_begin:node_begin + node_num[i]]
        node_begin=node_begin+node_num[i]
        node.append(torch.sum(node_slice,0))
    ast_node= torch.stack(node)
    return ast_node


class ClaModel(nn.Module):
    def __init__(self,
                 in_feature,
                 n_layers,
                 dropout,
                 n_class,
                 d_k=128, d_v=128, hidden_size=1024,device='cpu'):
        super(ClaModel,self).__init__()

        self.layers = nn.ModuleList(
            [HCLLayer(in_feature, d_k, d_v, hidden_size, dropout, device) for _ in range(n_layers)])
        # self.pre_model = pre_model
        self.pooling=pooling
        self.cla=nn.Sequential(
            nn.Linear(in_feature,1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,n_class),
        )
    def forward(self,g,node_emb,node_num,attn=None):
        for layer in self.layers:
            node_emb, attn = layer(g, node_emb, attn)

        ast=self.pooling(node_num,node_emb)
        cla_out=self.cla(ast)
        return cla_out


class CloModel(nn.Module):
    def __init__(self,
                 in_feature,
                 n_layers,
                 dropout,
                 n_class,
                 d_k=128, d_v=128, hidden_size=128,device='cuda'):
        super(CloModel,self).__init__()

        self.layers = nn.ModuleList(
            [HCLLayer(in_feature, d_k, d_v, hidden_size, dropout, device) for _ in range(n_layers)])
        # self.pre_model = pre_model
        self.pooling=pooling
        self.cla=nn.Sequential(
            nn.Linear(in_feature,128),
            nn.ReLU(),
            nn.Linear(128,n_class),
        )
    def forward(self,g1,node_emb1,path_emb1,node_num1,path_num1,g2,node_emb2,path_emb2,node_num2,path_num2,attn=None):
        for layer in self.layers:
            node_emb1, attn = layer(g1, node_emb1, attn)
        path_emb1=self.tf(path_emb1)
        ast_node1,ast_path1=self.pooling(node_num1,path_num1,node_emb1,path_emb1)
        ast_out1=self.gating(ast_node1,ast_node1,ast_path1)
        
        attn=None
        for layer in self.layers:
            node_emb2, attn = layer(g2, node_emb2, attn)
        path_emb2=self.tf(path_emb2)
        ast_node2,ast_path2=self.pooling(node_num2,path_num2,node_emb2,path_emb2)
        ast_out2=self.gating(ast_node2,ast_node2,ast_path2)

        ast_out= torch.abs(torch.add(ast_out1, -ast_out2))
        cla_out=self.cla(ast_out)
        cla_out = torch.sigmoid(cla_out)
        return cla_out

# '''
# 分类的测试
# '''
# import dgl
# srcs=[0,1,2,3,4]
# dsts=[1,0,1,4,5]
# g = dgl.graph((srcs, dsts))
# g = dgl.add_self_loop(g)
# # g = g.to('cuda:0')
# #
# node_emb=torch.randn(6,768)
# path_emb=torch.randn(3,768)
# node_num=[4,2]
# path_num=[1,2]
# # pre_model=torch.load('model.pkl')
# # pre_model_dict=pre_model.state_dict()
# model=ClaModel(768,4,4,0.5,2,device='cpu')
#
# ast_out=model(g,node_emb,node_num)
# print(ast_out)
#
#







