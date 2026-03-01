---
layout: post
author: Suhas Chundi
tags: [Transformer]
---

<details>
  <summary>Author's Notes</summary>
  
  We're continuing our implementation of a transformer from scratch. The transformer itself is quite basic, but computing the backwards gradients is a bit more involved.
</details>

Previous posts:
- [Part 1: SILU, Linear and RMSNorm]({% post_url 2025-09-17-Transformer-From-Scratch-Part-1 %})

The full code for my implementation is [here](https://github.com/s-chundi/from_scratch)

The standard for a successful implementation in this particular post will be "semi-coherent outputs from the model". We're training/testing with a subset of the cosmopedia_v2 dataset.

### The Embedding Layer

The forward pass for an embedding layer is a simple indexing operation. The backward pass requires collection of upstream gradients and assignment of those gradients to the embedding rows that are responsible for those gradients.

```python
class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, emb_matrix):
        """
        Args: 
            token_ids: longtensor of shape (B, S)
            emb_matrix: weight matrix of shape (n_vocab, D)
        Returns:
            tensor of shape B, S, D
        """
        ctx.save_for_backward(token_ids, emb_matrix)
        return emb_matrix[token_ids, :]

    @staticmethod
    def backward(ctx, dout):
        token_ids, emb_matrix = ctx.saved_tensors
        nv, D = emb_matrix.shape
        demb = None
        if ctx.needs_input_grad[1]:
            demb = torch.zeros_like(emb_matrix)
            grads_flattened = dout.reshape(-1, D)
            demb.index_add_(0, token_ids.reshape(-1), grads_flattened)
            
        return None, demb
```

### The Multi-Head Attention Mechanism

Multi head attention is mostly matrix multiplications, meaning its gradient derivation is not too complex as long as we are careful. The computation of the vector Jacobian product (VJP) for the softmax operation is derived here.

For a vector $x$, the softmax operation is given by:

$$\text{softmax}(x)_i = \frac{e^x_i}{\sum_{j} e^{x_j}}$$

Note every index $i$ is dependent on every other index $j$, so we must compute the Jacobian matrix.

$$J_{ij} = \frac{\partial \text{softmax}(x)_i}{\partial x_j}$$

For the Jacobian matrix, there are two cases to consider: $i = j$ and $i \neq j$. Let's also set $S = \sum_{k} e^{x_k}$.

For $i = j$, we have:

$$\frac{\partial \text{softmax}(x)_i}{\partial x_i} = \frac{\partial e^{x_i}}{\partial x_i} \cdot S^{-1} + e^{x_i} \cdot \frac{\partial S^{-1}}{\partial x_i}$$

$$ = \text{softmax}(x)_i  - \text{softmax}(x)_i \cdot \text{softmax}(x)_i$$

$$= \text{softmax}(x)_i  (1 - \text{softmax}(x)_i)$$

For $i \neq j$, we have:

$$\frac{\partial \text{softmax}(x)_i}{\partial x_j} = e^{x_i} * \frac{\partial S^{-1}}{\partial x_j} = e^{x_i} \cdot e^{-x_j} \cdot S^{-2} = -\text{softmax}(x)_i \cdot \text{softmax}(x)_j$$

We can concisely write this as `diag(softmax(x)) - softmax(x) @ softmax(x).T`

And instead of instantiating the full Jacobian matrix, we can compute the VJP by multiplying the incoming gradient vector by the Jacobian matrix.

`dout * softmax(x) - dout @ softmax(x) @ softmax(x).T`

The whole multi-head attention implementation is:

```python
class MHAFunction(torch.autograd.Function):
    @staticmethod
    def compute_attn_weights(qry, k, v, sequence_ids):
        attn_scores = einops.einsum(qry, k, "... sq nh d, ... sk nh d -> ... nh sq sk") / math.sqrt(k.shape[-1])
        causal_attn_mask = torch.triu(
            torch.ones(
                qry.shape[1],
                k.shape[1],
                dtype=torch.bool,
                device=k.device
            ),
            diagonal=k.shape[1] - qry.shape[1] + 1
        )
        packing_mask = sequence_ids[:, None, None, :] != sequence_ids[:, None, :, None]
        attn_scores.masked_fill_(packing_mask, -1e10)
        attn_scores.masked_fill_(causal_attn_mask, -1e10)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights
    
    @staticmethod
    def forward(ctx, qry, k, v, sequence_ids):
        """
        Args:
            Q, K, V with shape B, S, Nh, D_attn
            sequence_ids: longtensor of shape B, S
        Returns:
            tensor with shape B, S, Nh, D_attn
        """
        attn_weights = MHAFunction.compute_attn_weights(qry, k, v, sequence_ids)
        out = einops.einsum(attn_weights, v, "... nh sq sk, ... sk nh d -> ... sq nh d")
        
        ctx.save_for_backward(v, k, qry, sequence_ids)
        return out
    
    @staticmethod
    def backward(ctx, dout):
        v, k, qry, sequence_ids = ctx.saved_tensors
        attn_weights = MHAFunction.compute_attn_weights(qry, k, v, sequence_ids)
        
        dim = v.shape[-1]
        dqry, dk, dv, dkpm = None, None, None, None
        
        if ctx.needs_input_grad[2]:
            dv = einops.einsum(
                attn_weights, dout,
                "... nh sq sk, ... sq nh d -> ... sk nh d",
            )
            
        dattn_weights = einops.einsum(
            v, dout,
            "... sk nh d, ... sq nh d -> ... nh sq sk",
        )
        dlss = einops.einsum(
            attn_weights, dattn_weights, 
            "... sq sk, ... sq sk -> ... sq",
        ).unsqueeze(dim=-1)
        dattn_scores = attn_weights * (dattn_weights - dlss)
        if ctx.needs_input_grad[0]:
            dqry = einops.einsum(
                dattn_scores, k, 
                "... nh sq sk, ... sk nh d -> ... sq nh d"
            ) / math.sqrt(dim)
        if ctx.needs_input_grad[1]:
            dk = einops.einsum(
                dattn_scores, qry, 
                "... nh sq sk, ... sq nh d -> ... sk nh d"
            ) / math.sqrt(dim)
            
        return dqry, dk, dv, dkpm
```


### Generation Quality

<img src="/images/from-scratch/test_loss4.1.png">

At a test loss of ~4.1 (perplexity of ~60), the model performance holds sentence structure a little longer than our previous version, but the semantic coherence won't be great:
Results did improve from part 1, but not by much. I'm still training on subsections of cosmopedia_v2 (15k out of 39M samples), because I get impatient.

**Input:**
```
... contributions to fashion and art. Let's dive deeper into these aspects and learn what makes Paris so unique!

Paris Fashion – Dress Like a True Parisian!
----------------------------------------------

Imagine walking down the bustling streets of Paris, looking like a true Parisian woman! How do they achieve such elegant and sophisticated styles that are admired globally? It all starts with embracing classic yet contemporary clothing items.

### * Blazers & Shirts

Picture a neatly fitted navy blue blazer paired with a fresh, pressed white shirt – sounds simple, right? This dynamic duo forms the foundation of
```
**Output:**
```
a small group of your own graphic design! <|endoftext|>
```

Notice the continuation is a valid grammatical sentence, but it doesn't make much sense.