# Interpretability applied to MNIST

> ⚠️ ***WARNING, WARNING, WARNING*** (Voiced by Kevin Malone)  
>
> This repo is part of a live, interpretability-focused build process. It's updated regularly with new experiments, diagnostics, and reflections. Feedback is welcome.


## ❓ What is this Repo for ❓

This repo investigates how small neural networks trained on MNIST internally represent data — and how their behavior changes with normalization. The goal is to analyze activation dynamics, neuron specialization, and the effects of architectural and input changes using simple, interpretable setups.


## ✈️ Overview 
1) MLP experimentation (current task): We train a basic MLP on raw and normalized MNIST inputs, log internal states, track activation sparsity over training, and visualize how internal representations shift.
     - The first goal is to showcase the importance of normalization: we investigate activation mean/variance within the model, as well as neuron health, before and after normalization. Towards this we will cover techniques to probe the internals of the network, from simply logging activations in forward passes to implementing hooks for gradient tracking. We will also use activation heatmaps, and PCA to try and 'see' what the model is doing. 
     - Ultimately, one of the goals is to understand things commonly discussed in interpretability like polysemanticity, and superposition. Towards that we will investigate 
3) CNN experimentation (coming soon): We extend the same tools and analysis to a simple convolutional network. The goal is to compare how spatially-aware architectures differ from vectorized MLPs.

### Learning objectives  
This repo is intended to be both a learning tool and a educational resource. It walks through:
- How to build and probe a basic MLP on MNIST using PyTorch
- How neuron activations evolve during training
- How normalization impacts internal representations
- How to visualize dead neurons, representation drift, and layer dynamics
- More to come...



<!--
### 🥅 Goals
The goal is to turn this into a blog post or educational writeup on activations, dead neurons, and normalization effects in MLPs. This will be part of a broader interpretability portfolio including toy circuits and LLM probing.


## 💭 My hopes
- How to build an interpretable MLP (mirroring my [micrograd](/zero-to-hero-course/episode-1) architecture)
- How to track internal activations and neuron behaviors
- How to begin asking interpretability questions about hidden representations
If you’re early in your ML journey, or just looking to see how models can be dissected from first principles — I hope this is useful.
-->

## Progress tracking (Live Roadmap)

### MLP: 
- [✔️] Build a basic 'interpretable' MLP for experimentation

#### Baseline experimentation (without normalizing the MNIST data)
- [✔️] Train on MNIST (raw data)
- [✔️] log internal statistics during training (linear layers, activations, logits)
- [✔️] Add hooks to MLP
- [✔️] log internal states over training
- [✔️] Visualize training dynamics (loss curves, activation heatmaps)
- [✔️] Apply PCA to intermediate activations
- [ ] Investigate neuron specialization and representation drift
- [ ] Apply t-SNE to intermediate activations
- [ ] Apply UMAP ________ 

### Comparative experimentation: Normalized data and experiments
- [ ] Train on MNIST 
- [ ] Log internal states (activations, logits)
- [ ] Compare activation distributions across training (unnormalized vs normalized)
- [ ] Evaluate dead neuron rates and gradient flow
- [ ] Visualize how normalization impacts representation structure (via PCA)



### Mirror the experiments above with CNN
- [ ] Extend to CNN for comparison
- [ ] Prepare model and logs for deeper interpretability tools (e.g., TransformerLens-style analysis)




## Notebooks (Growing List)

| Notebook | Purpose |
|---------|---------|
| `01_MLP_for_Interpretability_non_normalized.ipynb` | Interpretable PyTorch MLP trained on raw MNIST input (hooks, dead neurons, histograms) |
| `02_MLP_for_Interpretability_normalized.ipynb` | Normalized-input training + PCA/activation comparisons |
| `03_CNN_for_Interpretability.ipynb`| (planned) Convolutional analog of the above, with same diagnostics|



## Why this❓❓

I learn by building things/working on projects: this repo reflects that. The goal is to **see what’s happening inside** of an MLP network trained on MNIST. I want to understand how even small networks form internal representations, and how we might design tools to probe, interpret, and improve them. This is one of many projects with an interpretability focus but I figured I would start here with the basics. 

If you have suggestions, corrections, or other feedback let me know!
  
