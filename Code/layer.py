from torch import nn
import torch
import time
import pytorch_lightning as pl
import numpy as np
import random
import torch_geometric
import os
from torch_geometric.nn import TransformerConv

def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]

class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.num_head = num_head
        self.att_layers = torch.nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(in_channel, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, 1, bias=False))
            )
        self.init_weight()
       
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Sequential):
                for sub_module in m:
                    if isinstance(sub_module, nn.Linear):
                        torch.nn.init.kaiming_uniform_(sub_module.weight.data)
                        if sub_module.bias is not None:
                            sub_module.bias.data.fill_(0.0)
    
    def forward(self, z):
        w = self.att_layers[0](z).mean(0)                    
        beta = torch.softmax(w, dim=0)                 
    
        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)
            
            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp
            
        return output / self.num_head

class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channels, out_channels, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        
        self.num_edge_type = num_edge_type
        self.transformer_list = torch.nn.ModuleList()
        # self.transformer_list.append(TransformerConv(in_channels=in_channels, out_channels=out_channels, heads=trans_heads, dropout=dropout, concat=False))
        for i in range(int(num_edge_type)):
            self.transformer_list.append(TransformerConv(in_channels=in_channels, out_channels=out_channels, heads=trans_heads, dropout=dropout, concat=False))
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channels + out_channels, in_channels),
            torch.nn.Sigmoid()
        )
        self.activation = torch.nn.ELU()
        self.semantic_attention = SemanticAttention(in_channel=out_channels, num_head=semantic_head)
       
        self.init_weight()
       
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Sequential):
                for sub_module in m:
                    if isinstance(sub_module, nn.Linear):
                        torch.nn.init.kaiming_uniform_(sub_module.weight.data)
                        if sub_module.bias is not None:
                            sub_module.bias.data.fill_(0.0)
        # self.transformer_list[0].reset_parameters()
        for i in range(self.num_edge_type):
            self.transformer_list[i].reset_parameters()

    def forward(self, features, edge_index, edge_type):
        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)
        
        u = self.transformer_list[0](features, edge_index_list[0].squeeze(0)).flatten(1) #.unsqueeze(1)
        a = self.gate(torch.cat((u, features), dim = 1))
        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))).unsqueeze(1)

        for i in range(1,len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i].squeeze(0)).flatten(1)
            a = self.gate(torch.cat((u, features), dim = 1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))
            semantic_embeddings=torch.cat((semantic_embeddings, output.unsqueeze(1)), dim = 1)
        
        return self.semantic_attention(semantic_embeddings)