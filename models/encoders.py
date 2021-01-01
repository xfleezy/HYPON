"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer,DenseAtt
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath
from torch.nn.modules.module import Module



class InvLinear(nn.Module):
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """
    def __init__(self, in_features, out_features,args):
        super(InvLinear, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.dim = self.in_features//self.out_features
        
        self.Gamma = nn.Linear(in_features,out_features)
        if self.args.use_mean:
            self.Lambda = nn.Linear(in_features//out_features,in_features//out_features)


    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """
        sets_alpha = nn.LeakyReLU(0.1, inplace=False)(self.Gamma(X))
        merged_feature = torch.zeros((X.shape[0],self.dim)).float().cuda()
        mean_feature = torch.zeros((X.shape[0],self.dim)).float().cuda()
        for i in range(self.out_features):
            merged_feature = merged_feature.clone() + sets_alpha[:, i:i + 1] * X[:, i * self.dim:(i + 1) *self.dim]
            mean_feature = merged_feature.clone() + X[:, i * self.dim:(i + 1) *self.dim]
        if self.args.use_mean:
            merged_feature = self.Lambda(merged_feature - mean_feature/(self.out_features))
        return merged_feature



class BidirectionalAgg(nn.Module):
    """
    Bidirectional aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att):
        super(BidirectionalAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        self.att_par = DenseAtt(in_features, dropout)
        self.att_chi = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        adj_att_par = self.att_par(x_par, adj)
        adj_child = adj.permute(1,0) 
        adj_att_chi = self.att_chi(x_tangent,adj_child)
        support_t = torch.matmul(adj_att_chi, x_tangent) +torch.matmul(adj_att_par, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)




class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class HYPON(Encoder):

    def __init__(self, c, args):
        super(HYPON, self).__init__(c)
        self.use_message = args.use_message
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.num = dims[0]//300
        self.bi_agg  = BidirectionalAgg(self.manifold,1,dims[0], args.dropout, args.use_att)
        self.inv_layer = InvLinear(dims[0],self.num,args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            if in_dim == dims[0]:
                  in_dim = 300
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj,adj_split):
        if self.use_message:
              x = self.bi_agg(x,adj_split)
        merged_feature = self.inv_layer(x)
        x_tan = self.manifold.proj_tan0(merged_feature, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HYPON, self).encode(x_hyp, adj)




class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
      
        self.curvatures.append(self.c)
        hgc_layers = []

        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
           
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj,_):
        
 
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)




