---
layout: post
author: Suhas Chundi
tags: [Mechanistic Interpretability]
---

Mechanistic Interpretability is the field of understanding how neural networks arrive at their outputs. Understanding language models is key to ensuring they are safe and aligned with human goals. There are many reasons LLMs are challenging to interpret. Internally, the patterns of activations of neurons cannot easily be correlated to certain behaviors. This is not intended to be an in depth research post. Instead I'm hoping to present some minimal code samples and interesting results for those hoping to get a peek at what mechanistic interpretability has to offer.

## The Superposition Hypothesisq

The Superposition Hypothesis is a common hypothesis for why LLM activations are so hard to understand. There are several orders of magnitude more concepts in the world than neurons in LLMs. Additionally, the concepts in the world are sparse; the word "pineapple" is in less than 0.1% of sentences, but still needs to be modeled by LLMs. Having a dedicated neuron for each concept is infeasible and unnecessary.

The figure below visualizes how a small model with 2 neurons can encode 5 features as sparsity increases. The lighter colored features are the most important. Notice that with no sparsity, the model ignores less important features and encodes the top 2 features orthogonally. As sparsity increases, the toy model finds a near orthogonal encoding of 5 features, and can use ReLU and bias terms to reduce noise.
<img src="/images/mech-int/superposition.png" />

The superposition hypothesis allows us to view LLMs as compressed versions of a larger LLM that is disentangled, interpretable, and sparse.
<img src="/images/mech-int/disentangled.png">
## SAEs

A natural course of action once understanding this is to disentangle an intermediate LLM activation. We can project an activation into a much higher dimensional space while imposing a sparsity penalty. This type of model is called a Sparse Autoencoder (SAE). Given an activation $h$ at some intermediate stage of the transformer, our SAE model setup is as follows:

$$z = ReLU(W_{enc}(h - b_{dec}) + b_{enc})$$

$$\hat{h} = W_{dec}z + b_{dec}$$

We would like to enforce a sparsity penalty on the high dimensional intermediate $z$, and a reconstruction loss between $\hat{h}$ and $h$. Setting $d$ as the dimension of the intermediate $z$, we have

$$L_{sparse} = ||z||_1$$

$$L_{reconstruction} = \frac{1}{d}||\hat{h} - {h}||_{2}^{2}$$

Now we can examine the (sparse) activations of $z$ and correlate them to concepts seen in inputs and outputs of models.

SAEs are (at the time of this writing) slightly dated, with newer architectures such as transcoders and crosscoders overcoming their limitations. This is also an oversimplified description of the SAE I actually trained, which uses a more sophisticated activation function (jumpReLU) to encourage binary features.

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

## Latent Exploration

There are several ways to analyze latents: brute force, [autointerp](https://blog.eleuther.ai/autointerp/), getting the top/bottom activations for a set of outputs. Here I'll show a more brute force approach, because I don't have the resources to run automated interpretability. We'll first visualize the activations

```
def show_activation_histogram(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    latent_idx: int,
    total_batches: int = 100,
):
    """
    Displays the activation histogram for a particular latent, computed across `total_batches` batches from `act_store`.
    """
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    all_positive_acts = []

    for i in tqdm(range(total_batches), desc="Computing activations for histogram"):
        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae_acts_post_hook_name],
        )
        acts = cache[sae_acts_post_hook_name][..., latent_idx]
        all_positive_acts.extend(acts[acts > 0].cpu().tolist())

    frac_active = len(all_positive_acts) / (total_batches * act_store.store_batch_size_prompts * act_store.context_size)
    
    max_act = max(all_positive_acts) if all_positive_acts else 0
    px.histogram(
        all_positive_acts,
        nbins=50,
        title=f"Latent {latent_idx} Activations Density {frac_active:.3%}, Max Activation {max_act:.2f}",
        labels={"value": "Activation"},
        width=800,
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
    ).update_layout(bargap=0.02, showlegend=False).show()


for i in random.sample(list(range(400)), 20):
    print(f"Density of latent {i}:")
    torch.cuda.empty_cache()
    show_activation_histogram(qwen, sae, activation_store, latent_idx=i)
```

SAE training is finicky. Some of the latents may be dead (never fire) as a result of the L1 penalty. This may be remedied with resampling techniques during training. Other latents may be dense, i.e. frequently active for diverse inputs, which suggests superposition is at play. The plot of such activations may look like this:
<img src="/images/mech-int/latent_381_dense.png">

Ideally, we see latents that rarely fire, as they correspond to only a single feature of the input text.
<img src="/images/mech-int/latent_10_sparse.png">

We can clamp these features high above their max activations to get some interesting and weird responses. This took a bit of manual labor to find interesting latents, as some are dense and others are not activated by the prompt or don't really produce interesting outputs. These responses don't imply a causal relationship between clamped latents and the outputs we see. But my objective with this work was to implement some research and produce some funny or strange results, which I think I've done.

<table class="outputs-table">
  <thead>
    <tr>
      <th>Steered Latent</th>
      <th>Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Normal output</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that have been asked for centuries. In the past, people used to think that identity was something fixed and unchanging. But now, we know that identity is more complex than that. It's not just...</td>
    </tr>
    <tr>
      <td>10</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that the artist asks in this series of works. The artist is interested in the idea of identity as a construct, something that is not fixed but rather something that is constantly being formed and reformed. The...</td>
    </tr>
    <tr>
      <td>10</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that the artist asks in her work. The artist is interested in the concept of identity and how it is constructed through the lens of the camera. She explores the idea of self through the act of photographing...</td>
    </tr>
    <tr>
      <td>10</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that the artist asks in this series of works. The artist is interested in the idea of identity as a construct, something that is not fixed but is instead fluid and ever-changing. The works explore the ways...</td>
    </tr>
    <tr>
      <td>29</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that the film "Identity" asks. The film is a psychological thriller that explores the concept of identity through the experiences of its characters. The story follows a group of strangers who are trapped in a remote cabin...</td>
    </tr>
    <tr>
      <td>29</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that the film "Identity" (2003) by David Fincher explores. The movie is a psychological thriller that delves into the complexities of identity and self-perception. In this article...</td>
    </tr>
    <tr>
      <td>29</td>
      <td>What are you? When you look in the mirror, what do you see? What is your identity? These are the questions that the film "Identity" asks. The film is about a group of people who are stranded in a cabin in the woods. Each of them has a different identity, and each has a different story...</td>
    </tr>
  </tbody>
</table>


```
from functools import partial
from rich import print as rprint
from rich.table import Table


GENERATE_KWARGS = dict(temperature=0.05, freq_penalty=2.0)

def steering_hook(
    activations: [Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    max_activation: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + torch.clamp(
        sae.W_dec[latent_idx] * max_activation * 10, 
        min=sae.W_dec[latent_idx] * max_activation * 5
    )
    
def generate_with_steering(
    model: HookedSAETransformer,
    sae: SAE,
    prompt: str,
    latent_idx: int,
    max_activation: float = 1.0,
    max_new_tokens: int = 50,
):
    """
    Generates text with steering. A multiple of the steering vector (the decoder weight for this latent) is added to
    the last sequence position before every forward pass.
    """
    _steering_hook = partial(
        steering_hook,
        sae=sae,
        latent_idx=latent_idx,
        max_activation=max_activation,
    )

    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, _steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

    return output


prompt = "What are you? When you look in the mirror, what do you see?"

no_steering_output = qwen.generate(prompt, max_new_tokens=50, **GENERATE_KWARGS)

table = Table(show_header=False, show_lines=True, title="Steering Output")
table.add_row("Normal", no_steering_output)
for latent, max_activation in [(10, 12.53), (29, 20.43)]:
    for i in tqdm(range(5), "Generating steered examples..."):
        table.add_row(
            f"Steered {latent} #{i}",
            generate_with_steering(
                qwen,
                sae,
                prompt,
                latent,
                max_activation=max_activation,
            ).replace("\n", "↵"),
        )
rprint(table)
```

## References

[SAELens](https://jbloomaus.github.io/SAELens/v5.10.7/)

[TransformerLens](https://transformerlensorg.github.io/TransformerLens/index.html)

[ARENA tutorials](https://arena-chapter1-transformer-interp.streamlit.app/)

[Transformer Circuits Thread](https://transformer-circuits.pub/)


