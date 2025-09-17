---
layout: post
author: Suhas Chundi
tags: [Transformer]
---

<details>
  <summary>Author's Notes</summary>
  
  Maintaining a blog is a lot of work. I'm going to try to transition to quick posts that take a couple days to write. Also I might sprinkle in some review for myself (such as this implementation) here and there. It seems like every blog now has now has an implementation of the transformer architecture. Even though my blog is barely 3 months old and is rarely updated, I don't want to be left out. 
</details>

The full code for my implementation is [here](https://github.com/s-chundi/from_scratch)

I'm not going to walk through the full transformer architecture, as other posts do a better job of that. In this quick post I will share sliding window attention and how it can speed up training.

## Sliding Window Attention

The strength of a language model is determined by how well it preserves context. The efficiency of a language model is determined by how well it compresses context. Transformers with full context attention are incredibly strong at modelling because the don't compress context at all. At each generation step, each token has access to every previous token. Consequently, they are not very efficient.

Sliding window attention is a naive way to compress context by attending to only the most recent $K$ tokens. Note the below implementation allows full context attention as well as sliding window attention by adjusting the $window\_size$ argument.

```
class CustomTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        window_size=0,
        n_head=4,
    ):
        super().__init__()
        assert embed_dim % n_head == 0
        self.ln1 = nn.LayerNorm(embed_dim)
        self.kv_linear = nn.Linear(embed_dim, embed_dim * 2)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
        self.window_size = window_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        
    def forward(self, x, key_pad_mask):
        ln1x = self.ln1(x)
        qry = self.q_linear(ln1x)
        qry = einops.rearrange(qry, "... sq (nh dattn) -> ... sq nh dattn", nh=self.n_head)
        kv = self.kv_linear(ln1x[:, -self.window_size:, :])
        kv = einops.rearrange(kv, "... sq (twoxnh dattn) -> ... sq twoxnh dattn", twoxnh=self.n_head*2)
        k, v = torch.chunk(kv, 2, dim=-2)
        attn_scores = einops.einsum(qry, k, "... sq nh d, ... sk nh d -> ... nh sq sk") / math.sqrt(x.shape[-1])
        causal_attn_mask = torch.triu(
            torch.ones(
                1,
                qry.shape[1],
                k.shape[1],
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )
        
        attn_scores.masked_fill_(key_pad_mask[:, None, None, -self.window_size:], -1e10)
        attn_scores.masked_fill_(causal_attn_mask, -1e10)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = einops.einsum(attn_weights, v, "... nh sq sk, ... sk nh d -> ... sq nh d")
        out = einops.rearrange(out, "... sq nh d -> ... sq (nh d)")
        return self.mlp(out) + x
```


I set window size to 2000 (1/4 of the original context window). Some preliminary timing tests (shown below) show the custom block is roughly two times faster. The Pytorch implementation of is probably much more efficient and doesn't compute masked values, hence the nonlinear tradeoff.

<img src="/images/from-scratch/fullattn.png">
<img src="/images/from-scratch/slidingattn.png">

I trained two transformers, each with 8 transformer blocks. In one, I alternate between sliding window attention and full attention, and in the other I only use full attention. The sliding window attention model outperforms the full attention model in both training time and test loss, which is probably a symptom of overfitting or needing better hyperparameters. The text corpus I used was Harry Potter and the Sorcerer's Stone.

<img src="/images/from-scratch/testloss.png">

## Generations

I also generate a few sequences from the models during validation to qualitatively check progress. The generations are not J.K. Rowling quality in the slightest, but they seem to be coherent english (up to a point). In future posts I will improve upon the quality and quantity of the training data, and set a higher bar for generation results than "somewhat english"

* ... You've had nearly fifteen minutes, now OUT" she said firmly. After a good night's sleep, **Harry felt nearly back to normal**

* ... As the door creaked, low, rumbling growls met their ears. All three of the dog's **noses that followed Ron** 

* ... Hermione's lip trembled, and she suddenly dashed at Harry and threw her arms around him. "Hermione!" "Harry -- you're a great **wizard, you know it, who was doing but it, who was doing but it, who was doing but it**