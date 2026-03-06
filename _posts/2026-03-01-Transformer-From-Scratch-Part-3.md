---
layout: post
author: Suhas Chundi
tags: [Transformer]
---

<details>
  <summary>Author's Notes</summary>
  
  In this post we'll implement KV caching and packing.
</details>

The full code for my implementation is [here](https://github.com/s-chundi/from_scratch)

Previous posts:
- [Part 2: Embedding Layer and Multi-Head Attention]({% post_url 2026-02-28-Transformer-From-Scratch-Part-2 %})
- [Part 1: SILU, Linear and RMSNorm]({% post_url 2025-09-17-Transformer-From-Scratch-Part-1 %})

### Packing

When training LLMs, input sequences have different lengths, but batches must have sequences of uniform length. The simplest approach to constructing a batch from variable length sequences is to pad shorter sequences with `<|endoftext|>` tokens. However, this means a significant portion of compute will be wasted on predicting `<|endoftext|>` tokens. 

An alternative approach is packing, where sequences are concatenated and delimited by `<|endoftext|>` tokens. This is fairly simple to implement in the data loader, but produces additional implementation requirements in the position embedding and attention computation steps.

### KV Caching

During the attention step, we produce a key, query, and value matrix with linear heads on the input. 
* Query vectors indicate "For this token, what other tokens should I consider?"
* Key vectors indicate "This token should be received later by tokens with certain properties."
* Value vectors store the any of the current token's properties that should be received by other tokens.

<details markdown="1">
  <summary>Side Note on Attention Circuits</summary>

  The attention equation $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$ can be interpreted as two circuits working together.
  * The QK circuit computes what items to pay attention to. The composition of linear heads $W_Q W_K$ determines how different inputs get routed / blended. However, it does not directly manipulate the embedding vectors.
  * The OV circuit manipulates embedding vectors directly. The composition of $W_V$ and the mlp after the attention step modifies embedding vectors for the next layer.
  * A more detailed explanation is given in the [transformers circuits thread](https://transformer-circuits.pub/).
</details>

Key and value vectors only need to be computed once during auto-regressive generation, as they are only dependent on tokens and layers that come before them. KV caching is the process of storing key and value vectors during generation to avoid recomputation and allow for more efficient inference.

Both of these optimizations require modifications primarily in the positional embedding step and the multi-head attention mechanism.

### Position Embedding

For the positional embedding, we have a mask which indicates the `<|endoftext|>` tokens, which for example could look like `[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]`, for which we need to generate a vector `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 5]` as input to the `pos_emb` layer. This is necessary so that each token's positional embedding is relative to its position in the original sequence, not the packed sequence. Since this is a "from scratch" implementation, I've opted to use low level `arange` and `cummax` functions.

Additionally, during generation, we will have only one token passed as input, but it still needs to be assigned the correct position embedding, meaning our transformer instance needs to track the current position during generation.

```python
class ScratchTransformer(nn.Module):
    def __init__(...):
        ...
        # Track current position during generation        
        self.cur_pos = 0
    
    def packing_helper(self, eottoken_mask):
        B, S = eottoken_mask.shape
        seq = torch.arange(1, S + 1, device=eottoken_mask.device).expand(B, S)
        offset = seq * eottoken_mask
        vals, inds = torch.cummax(offset, dim=1)
        sequence_ids = vals - offset
        return seq - sequence_ids - 1 + self.cur_pos, sequence_ids
    
    def forward(self, x, use_cache = False):
        """
        x : Tokens (B, S)
        """

        eottoken_mask = x == self.tokenizer.eot_token
        x = self.embed(x) # (B, S, D)
        pos_emb_input, sequence_ids = self.packing_helper(eottoken_mask)
        x = x + self.pos_emb(pos_emb_input)
        if use_cache:
            self.cur_pos += x.shape[-2]
        
        # Apply transformer blocks and unembedding
        
    def generate(self, x, num_tokens):
        assert num_tokens < self.context_win
        self.reset_cache()

        x = x[:, -self.context_win+num_tokens:]
        with torch.no_grad():
            # Populate the KV cache
            model_out, _ = self.forward(x, use_cache=True)
            _, inds = torch.max(model_out[:, -1, :], dim=1)
            x = torch.cat([x, inds.unsqueeze(-1)], dim=1)
            
            for _ in range(num_tokens - 1):
                # Passing in a single token is much more efficient than passing in the entire sequence
                model_out, __ = self.forward(x[:, -1:], use_cache=True)
                __, inds = torch.max(model_out[:, -1, :], dim=1)
                x = torch.cat([x, inds.unsqueeze(-1)], dim=1)
                
            # Decode the generated tokens
            
            self.reset_cache()
            return out_texts
        
    def reset_cache(self): # Reset KV cache for next generation
        self.cur_pos = 0
        for transformer in self.transformers:
            transformer.cache_k = None
            transformer.cache_v = None
```

### Multi-Head Attention

During the Multi-Head Attention step, tokens cannot attend to previous tokens that belong to other sequences. Hence we pass in a `sequence_ids` vector, which contributes to our causal attention mask. 

We also account for the case where the query shape is `(B, 1, D)` because we are auto regressively generating a single token at a time.

```python
class TransformerBlock(nn.Module):
    
    def __init__(...):
        ...
        
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        
    def forward(self, x, sequence_ids, use_cache):
        ...
        
        if use_cache:
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat((self.cache_k, k), dim=1)
                self.cache_v = torch.cat((self.cache_v, v), dim=1)
                k, v = self.cache_k, self.cache_v

        # k, v shapes are now (B, S, Nh, D)        
        out = MHAFunction.apply(qry, k, v, sequence_ids)
        ... # MLP and residual connection

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
            diagonal=k.shape[1] - qry.shape[1] + 1 # Works whether Query sequence length is 1 or S
        )
        packing_mask = sequence_ids[:, None, None, :] != sequence_ids[:, None, :, None] # Prevent tokens from attending to tokens from other sequences
        attn_scores.masked_fill_(packing_mask, -1e10)
        attn_scores.masked_fill_(causal_attn_mask, -1e10)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights

    @staticmethod
    def forward(ctx, qry, k, v, sequence_ids):
        """
        Args:
            Q, K, V with shape B, S, Nh, D_attn
            key_pad_mask will be used in the future
            
        Returns:
            tensor with shape B, S, Nh, D_attn
        """
        attn_weights = MHAFunction.compute_attn_weights(qry, k, v, sequence_ids)
        out = einops.einsum(attn_weights, v, "... nh sq sk, ... sk nh d -> ... sq nh d")
        
        ctx.save_for_backward(v, k, qry, sequence_ids)
        return out

    # Backward pass is unchanged
    
```