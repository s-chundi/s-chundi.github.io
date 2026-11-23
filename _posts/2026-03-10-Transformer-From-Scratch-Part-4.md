---
layout: post
author: Suhas Chundi
tags: [Transformer]
---

<details>
  <summary>Author's Notes</summary>
  
  Potentially the last post in the series. We're implementing RoPE and SWIGLU.
</details>

The full code for my implementation is [here](https://github.com/s-chundi/from_scratch)

Previous posts:
- [Part 3: KV Caching and Packing]({% post_url 2026-03-01-Transformer-From-Scratch-Part-3 %})
- [Part 2: Embedding Layer and Multi-Head Attention]({% post_url 2026-02-28-Transformer-From-Scratch-Part-2 %})
- [Part 1: SILU, Linear and RMSNorm]({% post_url 2025-09-17-Transformer-From-Scratch-Part-1 %})

### RoPE

Absolute positional embeddings are the simplest to implement, but face a couple of drawbacks. 
1. They do not generalize well to out of distribution sequence lengths. 
2. During packing, a very long sequence might get split across multiple context windows, corrupting the training process. 
3. They offer opportunities for models to overfit slightly on certain tokens being in certain positions.

RoPE is an effective positional encoding method that prioritizes the relative distance between tokens during the attention computation step.

The implementation is only a few additional lines of code.

```python
class TransformerBlock(nn.Module):
    
    def __init__(...):
        # other init steps
        head_dim = embed_dim // n_head
        angles = 1 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim)) # D_attn / 2
        seq_inds = torch.arange(CONTEXT_WINDOW)[:, None] # S, 1
        pos_angles = angles * seq_inds # S, D_attn / 2
        freqs = torch.polar(torch.ones_like(pos_angles), pos_angles)
        
        self.register_buffer("freqs", freqs, persistent=True)
    
    def rope_embed(self, x, pos_emb_input):
        """
        x : B, S, Nh, D
        pos_emb_input: B x [0, 1, 2, 0, 1, 2, ...] (S)
        """
        B, S, Nh, Da = x.shape

        x_complex = torch.view_as_complex(x.reshape(B, S, Nh, -1, 2))
        x_rotated = x_complex * self.freqs[pos_emb_input][:, :, None, :]
        
        x_out = torch.view_as_real(x_rotated).reshape(B, S, Nh, Da)
        
        return x_out
        
    def forward(self, x, sequence_ids, pos_emb_input, use_cache):
        # Key and query computations
        
        k = self.rope_embed(k, pos_emb_input)
        qry = self.rope_embed(qry, pos_emb_input)
        ...
```

Note the extent of rotation is based on both sequence index and dimension.
##### Rotation by Sequence Index
Suppose we have key vector $k_i$ and query vector $q_j$ at indices $i$ and $j$ respectively. 

After rotation, we have $k_i^* = k_i e^{i \theta_i}$ and $q_j^* = q_j e^{i \theta_j}$. The relative position between the rotated vectors is:

$$k_i^* {q_j^*}^T = k_i q_j e^{i (\theta_i - \theta_j)}$$

Thus the rotation by sequence index separates key and query vectors by relative position. 

##### Rotation Along Embedding Dimension
Keys and queries are rotated with different frequencies along the embedding dimension. As a result, the model can learn to place information that is position agnostic in later indices of the embedding dimension, and place information that is dependent on minute positional changes in the earlier dimensions. Technically, the rotation offers a non-monotonic relationship between position and embedding similarity, but in practice this does not significantly reduce performance.

### SWIGLU

The SWIGLU activation function is an upgrade over SILU (implemented in Part 1), and it shows performance improvements even when trainable parameters are controlled for. It is generally used in SOTA models, and combines the gating abilities of GLUs with the non-linearity of the SILU activation function:

$$y = W_1 x * \sigma(W_1 x) * W_2 x$$

Here's the gradient derivation (ignoring bias terms, but they're shown in the code):


$$u = W_1 x, \quad v = W_2 x$$  

$$\frac{\partial L}{\partial v} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial v} = \frac{\partial L}{\partial y} * u * \sigma(u)$$

$$\frac{\partial L}{\partial u} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial u} = \frac{\partial L}{\partial y} * v * \sigma(u) * (1 + u * (1 - \sigma(u)))$$

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial W_1}$$

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial W_2}$$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial x} + \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial x}$$

```python
class SWIGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, w1, b1, w2, b2):
        u = inp @ w1 + b1
        v = inp @ w2 + b2
        ctx.save_for_backward(inp, w1, b1, w2, b2)
        
        return u * torch.sigmoid(u) * v
    
    @staticmethod
    def backward(ctx, dout):
        inp, w1, b1, w2, b2 = ctx.saved_tensors
        
        u = inp @ w1 + b1
        sigmoid_u = torch.sigmoid(u)    
        v = inp @ w2 + b2
        
        dout_dsilu = dout * v
        dsilu_du = sigmoid_u * (1 + u * (1 - sigmoid_u))
        dout_du = dout_dsilu * dsilu_du
        dout_dv = dout * u * sigmoid_u
        
        dinp, dw1, db1, dw2, db2 = None, None, None, None, None
        if ctx.needs_input_grad[0]:
            dout_v_dinp = einops.einsum(
                dout_dv, w2, "... out, inp out -> ... inp"
            )
            dout_u_dinp = einops.einsum(
                dout_du, w1, "... out, inp out -> ... inp"
            )
            dinp = dout_v_dinp + dout_u_dinp
        if ctx.needs_input_grad[1]:
            dw1 = einops.einsum(
                dout_du, inp, "... out, ... inp -> inp out"
            )
        if ctx.needs_input_grad[2]:
            db1 = einops.reduce(
                dout_du, "... out -> 1 out", "sum"
            )
        if ctx.needs_input_grad[3]:
            dw2 = einops.einsum(
                dout_dv, inp, "... out, ... inp -> inp out"
            )
        if ctx.needs_input_grad[4]:
            db2 = einops.reduce(
                dout_dv, "... out -> 1 out", "sum"
            )
            
        return dinp, dw1, db1, dw2, db2
```