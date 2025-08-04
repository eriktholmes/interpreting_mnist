# Interpretability applied to MNIST

> ⚠️ ***WARNING, WARNING, WARNING*** (Voiced by Kevin Malone)  
>
> This is an active work-in-progress: a live repo documenting my ongoing interpretability experiments. Feedback is welcome and appreciated!

## ❓ What is this Repo

This repo investigates how small neural networks trained on MNIST internally represent data, and how their behavior shifts with normalization and architecture changes.

It is a pedagogical project aimed at:

- Building tools to probe internal model states (activations, gradients, neuron behavior)

- Visualizing training dynamics, e.g., dead neurons, sparsity, representation drift

- Exploring concepts like superposition and neuron polysemanticity — even in tiny MLPs

<br>

## ✈️ Overview 
### ✅ Phase 1: MLPs on Raw vs Normalized MNIST (in progress)
Trains a simple PyTorch MLP on MNIST with and without input normalization

Probes neuron activations during training (via hooks, histograms, PCA)

Tracks layer-wise statistics like activation variance, sparsity, and neuron "health"

Begins exploring early signs of polysemanticity and feature superposition

### 🧪 Coming Soon: CNN Interpretability
Apply the same techniques to a simple convolutional model

Compare spatial awareness vs dense MLP structure

Extend tools like activation heatmaps and PCA for conv layers

### 💭 Random thoughts: 
What happens if we train a Transformer-style architecture on MNIST?
> If we treat each image as a sequence of pixels (or patches), can attention mechanisms offer something between MLPs and CNNs — e.g., better inductive bias than a fully-connected net, but less spatial rigidity than a CNN?

### 📑 Learning objectives  
This repo is both a learning journal and educational toolkit. We will:

- Build & train interpretable MLPs in PyTorch
- Log activations & gradients manually and with hooks
- Visualize layer dynamics & neuron behavior
- Understand the role of normalization in network health
- Begin investigating interpretability concepts from Anthropic-style research


<br>

## Notebooks (Growing List)

| Notebook | Purpose |
|---------|---------|
| `01_MLP_for_Interpretability_non_normalized.ipynb` | Interpretable PyTorch MLP trained on raw MNIST input (hooks, dead neurons, histograms) |
| `02_MLP_for_Interpretability_normalized.ipynb` | Normalized-input training + PCA/activation comparisons |
| `03_CNN_for_Interpretability.ipynb`| (planned) Convolutional analog of the above, with same diagnostics|



## 🤔 Why MNIST?

MNIST is simple enough to fully analyze, yet rich enough to showcase key ideas and build/test mechanisms towards interpretability.

This project reflects how I learn best: by building, visualizing, and asking questions. It’s a first step in a larger interpretability journey — and a foundation for future experiments in Transformer internals and LLM behavior.

If you have suggestions, corrections, or other feedback let me know!
  
