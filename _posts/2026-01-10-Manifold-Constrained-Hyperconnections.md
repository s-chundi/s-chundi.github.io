---
layout: post
author: Suhas Chundi
tags: [Transformer]
---

<details>
  <summary>Author's Notes</summary>
  
  Happy New Year!
</details>

The full code for my implementation is [here](https://github.com/s-chundi/deepseek_mhc)

## Background

### The Residual Stream Tradeoff

The residual stream offers a way to treat each layer as an incremental adjustment to the input embeddings. At earlier layers, these adjustments or processing steps can focus on part of speech or syntax, and in later layers the incremental adjustments can focus on long-range semantics. 

##### Pre-norm residual connections
<div align="center">
$$x = x + Norm(F(x))$$
</div>

* The gradient will be $1 + (\epsilon) * \text{gradient}$, effectively handling the vanishing gradient problem, where $\epsilon$ is a small number b/c of the nature of layer normalization.
* The output of a layer $n$ will be $1 + \sum_{i=1}^{n} \epsilon * F_i(x)$, meaning all layers look very similar
* For stability purposes, this is the go-to for LLMs these days

##### Post-norm residual connections
<div align="center">
$$x = Norm(x + layer(x))$$
</div>

* The output of a layer is allowed to be very different than the input layer, adding feature diversity
* The gradient will be $\epsilon * (1 + \text{gradient})$, re-introducing the vanishing gradient problem
* This was used in the original transformers paper

### Enter Hyper Connections

<img src="/images/MCH/diagram.png">

This diagram offers a way to understand hyperconnections intuitively. Let $K$ be the residual stream "expansion". 

| Feature | Shape | Initialization |
|---------|-------|-----------------|
| Weighted Sum Vector | $(K,)$ | ones / K |
| Scaling Vector | $(K,)$ | ones |
| Mixing Matrix | $(K, K)$ | identity matrix |

* Note we are able to expand the residual stream with minimal additional parameters.
* As opposed to having shape $(D,)$ for the residual stream, it is now $(K, D / K)$.
* Small random noise is added during initialization to prevent identical gradients.

<details markdown="1">
  <summary>That is only static hyper connections. We can also have dynamic hyper connections.</summary>

Let $w$ be the matrix or vector for the weighted sum, mixing matrix, or scaling vector.
During the forward pass, the following transformation is applied:
<div align="center">
$$w = s_{\alpha} \cdot \tanh{(\text{Linear}(w))} + w$$
</div>
where $s_{\alpha}$ is a learned scalar. This allows the model to adjust hyper connections based on the input.
The paper also shows omitting the tanh function produces better results on some tasks. 

</details>
#### Sequential - Parallel Capabilities

A transformer model typically acts as a sequential model
$$x_{n+1}​=x_{n}​+F(x_{n}​)$$
* Early layers process POS, connect tokens into words, identify syntax
* Middle layers do high level semantic reasoning
* Late layers map the semantic understanding of the model to next token predictions

A transformer model can also (kind of) act as a parallel model. For example on a translation task,
* Layer 1 could extract POS info, and store this in indices 1 - 10 of the residual stream
* Layer 2 could extract gender and store this in indices 11 - 20 of the residual stream
* ...
* However this is sort of clunky b/c layer 2 has to understand and ignore the changes made by layer 1 to behave this way

This capability was made explicity in some old school open source models, (e.g. GPT-J) which used layers like:
<div align="center">
$$x_{out}​=x_{in}​+Attn(x_{in}​)+MLP(x_{in})$$
</div>
With the widened residual stream offered by hyper connections, the model can choose between sequential and parallel workflows very easily. We also see the model has greater variation between layers when using hyper connections.

<img src="/images/MCH/Cosine_sim_layers_hyperconnections.png" style="width: 50%;">

##### Results

<img src="/images/MCH/hyperconnection_loss_train.png" style="width: 65%;"> 

Loss curves for the OLMo model with and without hyper connections.

### Manifold Constrained Hyperconnections

##### Problem with Hyper Connections

<img src="/images/MCH/mhc_hc_baseling.png"> 

Each mixing matrix and scaling vector are unconstrained, allowing the model to arbitrarily amplify signals within the residual stream. While hyper connections proved useful in smaller models, they become unstable in larger models. 
*Note for the above experiment with 27B models, the hyper connection gradients fluctuate wildly.*

##### Constraints

The solution is to constrain the parameters manipulating the residual stream, preventing explosion in signal.

| Feature | Shape | Static Hyper Connection | Manifold Constraint Transformation |
|---------|-------|-------------------------|-------------------------------------|
| Weighted Sum Vector | $(K,)$ | $w$ | $\sigma(w)$
| Scaling Vector | $(K,)$ | $s$ | $2\sigma(s)$ |
| Mixing Matrix | $(K, K)$ | $M$ | $\text{Sinkhorn-Knopp}(e^{M})$ |

*The Sinkhorn-Knopp algorithm is just iterative row and column normalization of the matrix.*

##### Results

<img src="/images/MCH/mhc_benchmarks.png"> 

Benchmark results on a 27B model with and without manifold constrained hyper connections.

## Code

##### Baseline

We're using GSM-8k and Qwen3-0.6B
The first thing we'll do is evaluate the initial model on GSM-8k using ElutherAI's lm-eval package. We'll train the model on GSM-8k for 1 epoch and very low learning rate (`1e-6`) to get a baseline number.

For the baseline training, see the [baseline-train](https://github.com/s-chundi/deepseek_mhc/tree/baseline-train) branch.

You can follow along by following the quickstart in the [README](https://github.com/s-chundi/deepseek_mhc/blob/329a4e81805b273c0aee5ba71e93487ccfa2d3bb/README.md) at this checkpoint.
It is possible to take the model definition and configuration files of huggingface models, copy them into a project directory and point the transformers library to these files.

##### Vanilla Hyper Connections

We now modify the model to use vanilla hyper connections. Follow along in the code [here](https://github.com/s-chundi/deepseek_mhc/tree/hyperconnections) .

```python
hidden_states = torch.einsum(
                    "bksd,k->bsd",
                    hidden_states,
                    self.residual_stream_weights_attn
                )
# layernorm and attention 
hidden_states = torch.einsum(
                    "bsd,k->bksd",
                    hidden_states,
                    self.residual_stream_scaling_attn
                )
                residual = torch.einsum(
                    "bksd,kl->blsd",
                    residual,
                    self.residual_stream_mixing_attn
                )
                hidden_states = residual + hidden_states
                residual = hidden_states
```

For training setup and hyperparameters, see the [config.yaml](https://github.com/s-chundi/deepseek_mhc/blob/60622dcf70d1bf83788e3eda78336a203d3bfca7/src/model/config.yaml) for the branches `manifold-hyperconnections`, `hyperconnections` and `baseline-train`.

##### Manifold Constrained Hyper Connections
With some simple modifications, we can convert the model to manifold constrained hyper connections. 
```python
def sinkhorn_knopp(self, x):
    x = torch.exp(x)
    for _ in range(5):
        x = x / x.sum(dim=0, keepdim=True)
        x = x / x.sum(dim=1, keepdim=True)
        
    return x

def forward(
    self,
    hidden_states: torch.Tensor,
    ...
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = torch.einsum(
        "bksd,k->bsd",
        hidden_states,
        torch.sigmoid(self.residual_stream_weights_attn) # Notice sigmoid normalization
    )
    # layernorm and attention 
    hidden_states = torch.einsum(
        "bsd,k->bksd",
        hidden_states,
        torch.sigmoid(self.residual_stream_scaling_attn) * 2 # Again sigmoid normalization * 2
    )
    residual = torch.einsum(
        "bksd,kl->blsd",
        residual,
        self.sinkhorn_knopp(self.residual_stream_mixing_attn) # Sinkhorn-Knopp matrix normalization
    )
    hidden_states = residual + hidden_states
```

## Final Results

| Model  | gsm8K Score | Stderr |
|-------|-------|-------|
| Baseline | 0.4193 | 0.0136 |
| Vanilla Hyperconnections | **0.4496** | 0.0137 |
| Manifold Hyperconnections | 0.4147 | 0.0136 |


## Limitations
In order to use the original Qwen weights, we had to increase the size of the residual stream. It's possible that it was merely the extra computation that gave a boost in performance. In the original implementation, the residual stream footprint is unchanged (merely reshaped from $(D,)$ to $(K, D / K)$).

For training the baseline model, I did a grid search of hyperparameters, but it could be the case that better hyperparameters would have shown stronger performance.

## Conclusion

It's nice that we see a small improvement on gsm8K using hyperconnections. This aligns with the MCH paper that the manifold constraint is only effective for larger, deeper models.

