---
layout: post
author: Suhas Chundi
tags: [Mechanistic Interpretability]
---

Mechanistic Interpretability is the field of understanding how neural networks arrive at their outputs. Understanding Language models is key to ensuring they are safe and aligned with human goals. There are many reasons LLMs are challenging to interpret. Internally, the patterns of activations of neurons can not easily be correlated to certain behaviors. 

## The Superposition Hypothesis

The Superposition Hypothesis is common hypothesis for why LLM activations are so hard to understand. There are several orders of magnitude more concepts in the world than neurons in LLMs. Additionally, the concepts in the world are sparse; the word "pineapple" is in less than 0.1% of sentences, but still needs to be modeled by LLMs. Having a dedicated neuron for each concept is infeasible and unneccessary.

The figure below visualizes how a small model with 2 neurons can encode 5 features as sparsity increases. The lighter colored features are the most important. Notice that with no sparsity, the model ignores less important features and encodes the top 2 features orthogonally. As sparsity increases, the toy model finds a near orthogonal encoding of 5 features, and can use ReLU and bias terms to reduce noise.
<img src="/images/mech-int/superposition.png" />

The superposition hypothesis allows us to view LLMs as compressed versions of a larger LLM that is disentangled, interpretable, and sparse.
<img src="/images/mech-int/disentangled.png">
## SAEs

A natural course of action once understanding this is to disentangle an intermediate LLM activation! We can project an activation into a much higher dimensional space while imposing a sparsity penalty. This type of model is called a Sparse Autoencoder (SAE). Given an activation $h$ at some intermediate stage of the transformer, our SAE model setup is as follows:

$$z = ReLU(W_{enc}(h - b_{dec}) + b_{dec})$$

$$\hat{h} = W_{dec}z + b_{dec}$$

We would like to enforce a sparsity penalty on the high dimensional intermediate $z$, and a reconstruction loss between $\hat{h}$ and $h$. Setting $d$ as the dimension of the intermediate $z$, we have

$$L_{sparse} = ||z_i||_1$$

$$L_{reconstruction} = \frac{1}{d}||\hat{h} - {h}||_{2}^{2}$$

Now we can examine the (sparse) activations of $z$ and correlate them to concepts seen in inputs & outputs of models.

SAEs are (at the time of this writing) slightly dated, with newer architectures such as transcoders and crosscoders overcoming their limitations. This is also a oversimplified description of the SAE I actually trained, which uses a more sophisticated activation function (jumpReLU) to encourage binary features.

## Code

The fantastic [SAELens library](https://jbloomaus.github.io/SAELens/v5.10.7/) makes it very easy to train a SAE on activations from open source models. With only a few lines, I can train a SAE on the 28th layer of Qwen3-8b

```
import torch
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer = 28
runner_config = LanguageModelSAERunnerConfig(
    architecture="jumprelu",
    model_name="qwen3-8b",
    hook_name=f"blocks.{layer}.hook_mlp_out",
    hook_layer=layer,
    d_in=4096,
    dataset_path="HuggingFaceFW/fineweb",
    training_tokens=1228800000, # Will mainly determine training time
    prepend_bos=False,
    n_checkpoints=5,
    autocast=True,
    autocast_lm=True,
    wandb_project="s-chundi-SAE-qwen3-v1",
    device=device.type,
    seed=42,
    checkpoint_path="checkpoints",
)

sparse_autoencoder = SAETrainingRunner(runner_config).run() # Takes ~24 hours on a A100
sparse_autoencoder.save_model("final_checkpoint")
```
