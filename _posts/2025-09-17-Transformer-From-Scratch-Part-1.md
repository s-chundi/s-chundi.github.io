---
layout: post
author: Suhas Chundi
tags: [Transformer]
---

<details>
  <summary>Author's Notes</summary>
  
  Today I asked myself how annoyingly from scratch could I make a small transformer? Turns out, not that much. I can't write assembly and haven't written CUDA since my undergrad, but I think this is a good start. 
</details>

The full code for my implementation is [here](https://github.com/s-chundi/from_scratch)

I'm not going to walk through the full transformer architecture, as other posts do a better job of that. In Part 1 I'll implement everything except the multi-head attention mechanism, and hopefully I'll get to that in Part 2 someday.

The standard for a successful implementation in this particular post will be "coherent outputs from the model". We're training/testing with a subset of the cosmopedia_v2 dataset.

### A SILU activation function

The SILU activation function is given by:

$$y = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

To derive the gradient, we compute:

$$\frac{\partial y}{\partial x} = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))$$

Which is a very pretty gradient, as a result of the sigmoid function

Of course we're given $d_{out} = \frac{\partial L}{\partial y}$, so using the chain rule we compute:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left( \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x)) \right)$$

##### Code

```python
class SILUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp * 1 / (1 + torch.exp(-inp))
    
    @staticmethod
    def backward(ctx, dout):
        inp, = ctx.saved_tensors
        sigmoid_x = 1 / (1 + torch.exp(-inp))

        if ctx.needs_input_grad[0]:
            return dout * sigmoid_x * (1 + inp * (1 - sigmoid_x))
        else:
            return None
```

### RMSNorm

This gradient derivation is a bit more involved:

$$y = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 + \epsilon}}$$

Ignoring the $\epsilon$ term, we can set $\rho = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}$

$$\frac{\partial y}{\partial x} = \gamma \cdot \left( \frac{1}{\rho} - x \odot \partial (p^{-1}) \right) = \gamma \cdot \left( \frac{1}{\rho} - x \odot \frac{x}{\rho ^3 \cdot n} \right)$$

My notation might be a bit off there, but in code it looks like:

```python
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, gamma):
        rms = torch.sqrt(1e-8 + (inp ** 2).mean(dim=-1, keepdim=True))
        ctx.save_for_backward(inp, gamma, rms)
        return gamma * (inp / rms)
    
    @staticmethod
    def backward(ctx, dout):
        inp, gamma, rms = ctx.saved_tensors
        x_norm = inp / rms
        grad_x_norm = dout * gamma
        
        dinp, dgamma = None, None
        
        if ctx.needs_input_grad[0]:
            dinp = (
                grad_x_norm - x_norm * (x_norm * grad_x_norm).mean(dim=-1, keepdim=True)
            ) / rms
        if ctx.needs_input_grad[1]:
            dgamma = einops.reduce(dout * x_norm, "... embed_dim -> embed_dim", "sum")
            
        return dinp, dgamma
```

### A Linear layer

Finally, we'll wrap up with a simple linear layer. I don't think anybody who's read up to this point needs the gradient derivation.

```python
class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.save_for_backward(inp, weight)
        
        return inp @ weight + bias
    
    @staticmethod
    def backward(ctx, dout):
        inp, weight = ctx.saved_tensors
        
        din, dweight, dbias = None, None, None
        
        if ctx.needs_input_grad[0]:
            din = einops.einsum(
                dout, weight, "... out, inp out -> ... inp"
            )
        if ctx.needs_input_grad[1]:
            dweight = einops.einsum(
                dout, inp, "... out, ... inp -> inp out"
            )
        if ctx.needs_input_grad[2]:
            dbias = einops.reduce(dout, "... out -> 1 out", "sum")
            
        return din, dweight, dbias
```

Note, we use the check `ctx.needs_input_grad` in case some layers are frozen in the future.

### Sanity Check

We can train our model and see if the generations are coherent.

**Model Input:**
```
... each step and how it helps achieve the desired outcome, inluding key tips and guidelines. Ensure clarity and practicality, allowing readers to easily follow and apply the instructions. Do not use images.  Title: How to Be an Exotic Dancer

Exotic dancing is a form of dance that involves performing sensual moves in front of an audience, often in a nightclub or strip club setting. This guide will provide you with comprehensive steps on how to become an exotic dancer, as well as important considerations and safety measures. Please note that this career requires physical stamina, self-confidence,
```
**Output:**
```
navigating the importance of the power and maintaining good contact with the magic behind this chapter, including those with friends but they're going to improve social cohesion
```

Looks like english but pretty semantically rambling. I trained on a small subset of the dataset and didn't train for too many epochs, hence the poor generation quality. I'll circle back to improving generations in Part 2.