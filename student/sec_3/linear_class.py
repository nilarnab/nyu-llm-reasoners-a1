import math
from typing import Any
import torch
from numpy.random import normal
from torch import Tensor
from einops import rearrange
from jaxtyping import Bool, Float, Int


from tests.conftest import d_model


def init_weights(in_features, out_features, device):
    weight_tensor = torch.empty(in_features, out_features)

    if device is not None:
        weight_tensor = weight_tensor.to(device=device)

    sigma = (2 / (out_features + in_features)) ** 0.5
    torch.nn.init.trunc_normal_(weight_tensor, 0, sigma, -3 * sigma, 3 * sigma)

    return torch.nn.Parameter(weight_tensor)

class Linear(torch.nn.Module):
    def __init__(self, in_features : int, out_features: int, weights=None, device=None, dtype=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.weights = self._init_weights(weights)
        self.device = device
        self.dtype = dtype

    def _init_weights(self, weight_tensor : torch.Tensor = None):
        if weight_tensor is None:
            weight_tensor = torch.randn(self.out_features, self.in_features)
            sigma = (2 / (self.out_features + self.in_features)) ** 0.5

            torch.nn.init.trunc_normal_(weight_tensor, 0, sigma, -3*sigma, 3*sigma)

        return torch.nn.Parameter(weight_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("weight shape", self.weights.shape)
        # print("feature shape", x.shape)
        W = rearrange(self.weights, "o d -> d o")
        # print("weight shape", W.shape)
        res = x @ W

        # print("result shape", res.shape)

        return res

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, weights=None, device=None, dtype=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = None
        self.device = device
        self.dtype = dtype
        self.weights = self._init_weights(weights)

    def _init_weights(self, weight_tensor : torch.Tensor = None):
        if weight_tensor is None:
            weight_tensor = torch.randn(
                self.num_embeddings,
                self.embedding_dim,
                device=self.device)
            sigma = (2 / (self.num_embeddings + self.embedding_dim)) ** 0.5

            torch.nn.init.trunc_normal_(weight_tensor, 0, sigma, -3*sigma, 3*sigma)

        return torch.nn.Parameter(weight_tensor)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # print('weight shape', self.weights.shape)
        # print("num embeddings", self.num_embeddings)
        # print("embedding dim", self.embedding_dim)
        # print('token idds shape', token_ids.shape)
        # print("token ids", token_ids)
        token_ids.to(self.weights.device)

        return self.weights[token_ids]

        # return None


class RMSNorm(torch.nn.Module):
    #TODO: check the theory again
    def __init__(self, d_model: int, eps: float = 1e-5, weights=None,device=None, dtype=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = self._init_weights(weights)

    def _init_weights(self, weights):
        if weights is None:
            return torch.nn.Parameter(torch.ones(self.d_model, device=self.device)) #TODO: have to review how it is done
        return torch.nn.Parameter(weights)

    def forward(self, x : torch.Tensor):
        # print("x shape", x.shape)
        # print("weights shape", self.weights.shape)

        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_2 = x*x
        x_2_sum = x_2.sum()

        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps) # TODO: understand this one
        # print("rms shape", rms.shape)

        rms_norm = (x / rms) * self.weights

        # print("x shape", x.shape)

        # rms_x = ()**0.5

        return rms_norm

        return None


def silu(x):
    # print("silu shapes", x.shape, torch.sigmoid(x).shape)
    return x * torch.sigmoid(x)

class PositionFeedForwardSwigLu(torch.nn.Module):
    # TODO: learn the different shapes
    def __init__(self, d_model, d_ff, w1_weight=None, w2_weight=None, w3_weight=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = w1_weight if w1_weight is not None else init_weights(d_ff, d_model, device)
        self.w2_weight = w2_weight if w2_weight is not None else init_weights(d_model, d_ff, device)
        self.w3_weight = w3_weight if w3_weight is not None else init_weights(d_ff, d_model, device)
        self.use_silu = False

    def forward(self, in_features):

        # print("shape in_feature", in_features.shape)
        # print("shape w 1", self.w1_weight.shape)
        # print("shape w 2", self.w2_weight.shape)
        # print("shape w 3", self.w3_weight.shape)
        # print("d_model", self.d_model, "d_ff", self.d_ff)
        if not self.use_silu:
            silu_w1_x = silu(in_features @ rearrange(self.w1_weight, "o i -> i o"))
            w3_x = in_features @ rearrange(self.w3_weight, "o i -> i o")
            mult_val = silu_w1_x * w3_x

            # print("shape", mult_val.shape)
            res = mult_val @ rearrange(self.w2_weight, "o i -> i o")

            return res
        else:
            x = in_features @ rearrange(self.w1_weight, "o i -> i o")
            x = silu(x)
            x = x @ rearrange(self.w2_weight, "o i -> i o")
            return x


def run_softmax_util(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # TODO: understand the "along dimths axis" and how is differnth than just
    # taking in_features[dim], understand keepdim true meaning

    tensor_inp = in_features

    max_val, _ = torch.max(tensor_inp,dim=dim, keepdim=True) #TODO: understand the output of max function
    # # print("max val", max_val)

    tensor_inp = tensor_inp - max_val

    exp_sm = torch.exp(tensor_inp).sum(dim=dim, keepdim=True) #TODO: understand broadcasting

    tensor_inp = torch.exp(tensor_inp) / exp_sm


    return tensor_inp

    return None


class RotaryPositionalEmbedding:
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        token_positions = token_positions.to(self.device)

        THETA = self.theta
        x_pair = rearrange(x,'... (c d) -> ... c d', d=2)
        pair_pos = torch.arange(x_pair.shape[-2], device=self.device)

        freqs = (1.0/ (THETA ** (2*pair_pos/x.shape[-1]))).to(self.device)

        if token_positions.dim() == 1:
            token_positions = rearrange(token_positions, 'seq -> 1 seq 1')
        else:
            token_positions = rearrange(token_positions, 'batch seq -> batch seq 1')

        freqs = rearrange(freqs, 'd -> 1 1 d')
        theta = token_positions * freqs

        sins = rearrange(theta.sin(), 'b seq d -> b seq d 1')
        coses = rearrange(theta.cos(), 'b seq d -> b seq d 1')

        x1 = x_pair[..., 0:1]
        x2 = x_pair[..., 1:2]

        x_rotated = torch.cat([x1 * coses - x2 * sins,
                               x1 * sins + x2 * coses], dim=-1)

        x_out = rearrange(x_rotated, '... c d -> ... (c d)')

        return x_out

def scaled_dot_product_attention_util(
        Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # print("")
    # print("shapes Q", Q.shape)
    # print("shapes K", K.shape)
    # print("shapes V", V.shape)
    # print("shape mask", mask.shape)

    K_transpose = rearrange(K, "... l d -> ... d l")
    # print("q transpose shape", K_transpose.shape)

    d_k = Q.shape[-1]
    qt_k_mult = ( Q @ K_transpose)

    mask_tensor = torch.where(
        mask,
        torch.tensor(0.0, device=mask.device),
        torch.tensor(-float('inf'), device=mask.device)
    )

    # print("qt_k_mult shape", qt_k_mult.shape)
    attention_val = run_softmax_util((qt_k_mult + mask_tensor)/d_k ** 0.5, -1) @ V
    # print("attention val shape", attention_val.shape)


    return attention_val



class MultiHeadedAttentionRoped(torch.nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 max_seq_len: int,
                 theta: float,
                 q_proj_weight = None,
                 k_proj_weight = None,
                 v_proj_weight = None,
                 o_proj_weight = None,
                 device: str = None,
                 use_pe = True
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.d_k = d_model // num_heads
        self.q_proj_weight = q_proj_weight if q_proj_weight is not None else init_weights(d_model, d_model, device)
        self.k_proj_weight = k_proj_weight if k_proj_weight is not None else init_weights(d_model, d_model, device)
        self.v_proj_weight = v_proj_weight if v_proj_weight is not None else init_weights(d_model, d_model, device)
        self.o_proj_weight = o_proj_weight if o_proj_weight is not None else init_weights(d_model, d_model, device)

        self.q_proj_weight_T = rearrange(self.q_proj_weight, "... a b -> ... b a")
        self.k_proj_weight_T = rearrange(self.k_proj_weight, "... a b -> ... b a")
        self.v_proj_weight_T = rearrange(self.v_proj_weight, "... a b -> ... b a")
        self.o_proj_weight_T = rearrange(self.o_proj_weight, "... a b -> ... b a")

        d_head = self.d_model // self.num_heads
        self.rope_layer = RotaryPositionalEmbedding(self.theta, d_head, self.max_seq_len, device=self.device)
        self.use_pe = use_pe


    def forward(self,
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:

        if token_positions is None:
            token_positions = torch.arange(
                in_features.shape[-2],
                device=in_features.device
            )

        # print("")
        # print("d_model", self.d_model)
        # print("num_heads", self.num_heads)
        # print("q_proj_weight", self.q_proj_weight.shape)
        # print("k_proj_weight", self.k_proj_weight.shape)
        # print("v_proj_weight", self.v_proj_weight.shape)
        # print("o_proj_weight", self.o_proj_weight.shape)
        # print("in_features", in_features.shape)

        Q = in_features @ self.q_proj_weight_T
        K = in_features @ self.k_proj_weight_T
        V = in_features @ self.v_proj_weight_T
        O = in_features @ self.o_proj_weight_T

        # Q = rope_layer.forward(Q, token_positions)
        # K = rope_layer.forward(K, token_positions)

        # print("ropee applied")

        # q_split = rearrange(Q, "... (h b) -> h ... b", h=num_heads)
        # k_split = rearrange(K, "... (h b) -> h ... b", h=num_heads)
        # v_split = rearrange(V, "... (h b) -> h ... b", h=num_heads)
        q_split = rearrange(Q, "b t (h d_h) -> b h t d_h", h=self.num_heads)  # TODO: why this is working
        k_split = rearrange(K, "b t (h d_h) -> b h t d_h", h=self.num_heads)
        v_split = rearrange(V, "b t (h d_h) -> b h t d_h", h=self.num_heads)
        # print("q_split shpae", q_split.shape)

          # Use head dimension, not full d_model!

        if self.use_pe:
            q_split = self.rope_layer.forward(q_split, token_positions)
            k_split = self.rope_layer.forward(k_split, token_positions)
        # print("RoPE applied with d_head =", d_head)

        seq_len = in_features.shape[-2]
        # mask_causal = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool()
        mask_causal = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=0).bool()

        # all_heads_res = []

        # for head_i in range(self.num_heads):
        #     # print("head i", head_i)
        #     # TODO: understand this breaking
        #     q = q_split[:, head_i, :, :]
        #     k = k_split[:, head_i, :, :]
        #     v = v_split[:, head_i, :, :]
        #
        #     # mask_uncausal = torch.full((q @ rearrange(k, "... l d -> ... d l")).shape, 1, device=q_proj_weight.device)
        #     # mask_causal = torch.triu(mask_uncausal, diagonal=1).bool()
        #
        #     # print("mask causal shape", mask_causal.shape)
        #
        #     attention_val = scaled_dot_product_attention_util(q, k, v, mask_causal)
        #
        #     # print("multiheaded attention val shape", attention_val.shape)
        #
        #     all_heads_res.append(attention_val)

        B, H, T, D_h = q_split.shape
        q_split = q_split.reshape(B * H, T, D_h)
        k_split = k_split.reshape(B * H, T, D_h)
        v_split = v_split.reshape(B * H, T, D_h)

        mask_causal_expanded = mask_causal.unsqueeze(0).expand(B * H, T, T)

        concat_res = scaled_dot_product_attention_util(q_split, k_split, v_split, mask_causal_expanded)
        concat_res = concat_res.reshape(B, H, T, D_h)
        concat_res = concat_res.permute(0, 2, 1, 3).reshape(B, T, H * D_h)

        res = concat_res @ self.o_proj_weight_T
        # print(res)

        return res


class MultiHeadedAttentionNonRoped(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self,
                q_proj_weight: Float[Tensor, " d_k d_in"],
                k_proj_weight: Float[Tensor, " d_k d_in"],
                v_proj_weight: Float[Tensor, " d_v d_in"],
                o_proj_weight: Float[Tensor, " d_model d_v"],
                in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
        # print("")
        # print("d_model", self.d_model)
        # print("num_heads", self.num_heads)
        # print("q_proj_weight", q_proj_weight.shape)
        # print("k_proj_weight", k_proj_weight.shape)
        # print("v_proj_weight", v_proj_weight.shape)
        # print("o_proj_weight", o_proj_weight.shape)
        # print("in_features", in_features.shape)

        Q = in_features @ rearrange(q_proj_weight, "... a b -> ... b a")
        K = in_features @ rearrange(k_proj_weight, "... a b -> ... b a")
        V = in_features @ rearrange(v_proj_weight, "... a b -> ... b a")
        O = in_features @ rearrange(o_proj_weight, "... a b -> ... b a")

        # q_split = rearrange(Q, "... (h b) -> h ... b", h=num_heads)
        # k_split = rearrange(K, "... (h b) -> h ... b", h=num_heads)
        # v_split = rearrange(V, "... (h b) -> h ... b", h=num_heads)
        q_split = rearrange(Q, "b t (h d_h) -> b h t d_h", h=self.num_heads)  # TODO: why this is working
        k_split = rearrange(K, "b t (h d_h) -> b h t d_h", h=self.num_heads)
        v_split = rearrange(V, "b t (h d_h) -> b h t d_h", h=self.num_heads)
        # print("q_split shpae", q_split.shape)

        seq_len = in_features.shape[-2]
        # mask_causal = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool()
        mask_causal = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=0).bool()

        all_heads_res = []

        for head_i in range(self.num_heads):
            # print("head i", head_i)
            # TODO: understand this breaking
            q = q_split[:, head_i, :, :]
            k = k_split[:, head_i, :, :]
            v = v_split[:, head_i, :, :]

            # mask_uncausal = torch.full((q @ rearrange(k, "... l d -> ... d l")).shape, 1, device=q_proj_weight.device)
            # mask_causal = torch.triu(mask_uncausal, diagonal=1).bool()

            # print("mask causal shape", mask_causal.shape)

            attention_val = scaled_dot_product_attention_util(q, k, v, mask_causal)

            # print("multiheaded attention val shape", attention_val.shape)

            all_heads_res.append(attention_val)

        concat_res = torch.cat(all_heads_res, dim=-1)
        # # print(concat_res)

        # print('concat res shape', concat_res.shape)

        # print("O shape", O.shape)

        res = concat_res @ o_proj_weight.T

        # print(res)

        return res


class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 max_seq_len,
                 theta,
                 rms1_weight = None,
                 rms2_weight = None,
                 pffs_weight1 = None,
                 pffs_weight2 = None,
                 pffs_weight3 = None,
                 q_proj_weight = None,
                 k_proj_weight = None,
                 v_proj_weight = None,
                 o_proj_weight = None,
                 device=None,
                 post_norm=False,
                 use_rms=True,
                 use_pe=True,
                 use_silu=False,
                 ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.post_norm = post_norm
        self.use_silu = use_silu
        self.rms_layer_1 = RMSNorm(d_model, 1e-5, rms1_weight,device=self.device)
        self.rms_layer_2 = RMSNorm(d_model, 1e-5, rms2_weight,device=self.device)
        self.mha_layer = MultiHeadedAttentionRoped(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,device=self.device)
        self.pffs_layer = PositionFeedForwardSwigLu(d_model, d_ff, pffs_weight1, pffs_weight2, pffs_weight3,device=self.device,use_silu=self.use_silu)
        self.use_rms = use_rms
        self.use_pe = use_pe
        self.mha_layer = MultiHeadedAttentionRoped(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight,
                                                   v_proj_weight, o_proj_weight, device=self.device, use_pe=self.use_pe)

    def forward(self, in_features):

        if not self.post_norm: # pre norm
            in_features = in_features.to(self.device)
            if self.use_rms:
                rms_val = self.rms_layer_1.forward(in_features)
                mha_roped_val = self.mha_layer.forward(rms_val)
            else:
                mha_roped_val = self.mha_layer.forward(in_features)

            y = in_features + mha_roped_val
            # print("y shape", y.shape)
            if self.use_rms:
                rms2_val = self.rms_layer_2.forward(y)
                pffs_val = self.pffs_layer.forward(rms2_val)
            else:
                pffs_val = self.pffs_layer.forward(y)


            y_2 = y + pffs_val
            # print("y_2 shape", y_2.shape)
        else:
            in_features = in_features.to(self.device)

            mha_val = self.mha_layer.forward(in_features)
            if self.use_rms:
                mha_val = self.rms_layer_1.forward(mha_val)

            y = in_features + mha_val

            pffs_val = self.pffs_layer.forward(y)
            if self.use_rms:
                pffs_val = self.rms_layer_2.forward(pffs_val)
            y_2 = y + pffs_val

        return y_2


class TransformerLm(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor] = None,
                 device='cpu',
                 post_norm = False,
                 use_pe = True,
                 use_rms = True,
                 use_silu = False
                 ):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.state_dir = weights
        self.device = device
        self.post_norm = post_norm
        self.use_silu = use_silu
        self.use_pe = use_pe
        self.use_rms = use_rms

        # layers present
        self.embedding_layer = Embedding(vocab_size, d_model, self.state_dir["token_embeddings.weight"] if weights is not None else None)
        self.transformer_blocks = [
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                rms1_weight = self.state_dir[f"layers.{i}.ln1.weight"] if weights is not None else None,
                rms2_weight=self.state_dir[f"layers.{i}.ln2.weight"] if weights is not None else None,
                pffs_weight1=self.state_dir[f'layers.{i}.ffn.w1.weight'] if weights is not None else None,
                pffs_weight2=self.state_dir[f'layers.{i}.ffn.w2.weight'] if weights is not None else None,
                pffs_weight3=self.state_dir[f'layers.{i}.ffn.w3.weight'] if weights is not None else None,
                q_proj_weight=self.state_dir[f'layers.{i}.attn.q_proj.weight'] if weights is not None else None,
                k_proj_weight=self.state_dir[f'layers.{i}.attn.k_proj.weight'] if weights is not None else None,
                v_proj_weight=self.state_dir[f'layers.{i}.attn.v_proj.weight'] if weights is not None else None,
                o_proj_weight=self.state_dir[f'layers.{i}.attn.output_proj.weight'] if weights is not None else None,
                device=self.device,
                post_norm=self.post_norm,
                use_pe=self.use_pe,
                use_rms=self.use_rms,
                use_silu=self.use_silu
            )
            for i in range(num_layers)
        ]
        self.rms_final_layer = RMSNorm(d_model, 1e-5, self.state_dir[f"ln_final.weight"] if weights is not None else None)
        self.linear_layer = Linear(d_model, vocab_size, weights=self.state_dir["lm_head.weight"] if weights is not None else None) # TODO: how to find out its shape in advance


    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        # print("in_indices shape", in_indices.shape)
        embedding_val = self.embedding_layer.forward(in_indices.to(self.device))

        transformer_val = embedding_val
        for layer in self.transformer_blocks:
            transformer_val = layer.forward(transformer_val)

        norm_val = self.rms_final_layer.forward(transformer_val)
        linear_val = self.linear_layer(norm_val)

        # TODO: why on earth softmax not to be used
        # res = run_softmax_util(linear_val, dim=-1)
        res = linear_val
        # print("res shape", res.shape)
        # print("softmax shape",run_softmax_util(linear_val, dim=-1).shape)

        return res


# TODO: BRUSH up the math behind it again, why taking softmax and then log is not wokring
def run_log_softmax_util(
    in_features: Float[Tensor, "..."],
    dim: int
) -> Float[Tensor, "..."]:

    max_val, _ = torch.max(in_features, dim=dim, keepdim=True)
    shifted = in_features - max_val

    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))

    return shifted - log_sum_exp


# SECTION 4
# ==============




# ==========


def run_multihead_self_attention_util(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    mhal_nonroped = MultiHeadedAttentionNonRoped(d_model, num_heads)
    res = mhal_nonroped.forward(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features)

    return res


def run_multihead_self_attention_with_rope_util(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:

    mhal_roped = MultiHeadedAttentionRoped(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)

    res = mhal_roped.forward(in_features, token_positions)

    return res
    return None









