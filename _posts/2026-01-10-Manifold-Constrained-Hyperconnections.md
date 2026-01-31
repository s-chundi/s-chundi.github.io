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
* The output of a layer $n$ will be $1 + \sum_{i=1}^{n} \epsilon$, meaning all layers look very similar
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
| Weighted Sum Vector | $(K,)$ | one hot |
| Scaling Vector | $(K,)$ | ones |
| Mixing Matrix | $(K, K)$ | identity matrix |

* Note we are able to expand the residual stream with minimal additional parameters.
* As opposed to having shape $(D,)$ for the residual stream, it is now $(K, D / K)$.

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
With the widened residual stream offered by hyper connections, the model can choose between sequential and parallel workflows very easily.

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

My code up to this point is checkpointed [here](https://github.com/s-chundi/deepseek_mhc/blob/329a4e81805b273c0aee5ba71e93487ccfa2d3bb)

You can follow along by following the quickstart in the [README](https://github.com/s-chundi/deepseek_mhc/blob/329a4e81805b273c0aee5ba71e93487ccfa2d3bb/README.md) at this checkpoint.

```
============================================================
GSM8K Results (num_fewshot = 5)
============================================================
alias: gsm8k
exact_match,strict-match: 0.4109
exact_match_stderr,strict-match: 0.0136
exact_match,flexible-extract: 0.4109
exact_match_stderr,flexible-extract: 0.0136
```

If we do a full SFT run, we get even worse results:

```
============================================================
GSM8K Results  (num_fewshot = 5)
============================================================
alias: gsm8k
exact_match,strict-match: 0.3624
exact_match_stderr,strict-match: 0.0132
exact_match,flexible-extract: 0.3639
exact_match_stderr,flexible-extract: 0.0133
```
##### Vanilla Hyper Connections

We now modify the model to use vanilla hyper connections. Follow along in the code [here](https://github.com/s-chundi/deepseek_mhc/tree/hyperconnections) .

After SFT, we get some minor improvements on GSM-8k.

```
============================================================
GSM8K Results (num_fewshot = 5)
============================================================
alias: gsm8k
exact_match,strict-match: 0.4443
exact_match_stderr,strict-match: 0.0137
exact_match,flexible-extract: 0.4450
exact_match_stderr,flexible-extract: 0.0137
```

Then we can do some post-training with GRPO

```
============================================================
GSM8K Results (num_fewshot = 5)
============================================================
alias: gsm8k
exact_match,strict-match: 0.4541
exact_match_stderr,strict-match: 0.0137
exact_match,flexible-extract: 0.4549
exact_match_stderr,flexible-extract: 0.0137
```