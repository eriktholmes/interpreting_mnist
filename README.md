# Interpretability applied to MNIST

⚠️ ***WARNING, WARNING, WARNING*** (Voiced by Kevin Malone) 

**This is a work in progress...**  This notebook is part of a live, educational build as I learn and explore interpretability tools from the ground up. It will be updated regularly with new experiments, edits, and insights. Suggestions and issues are always welcome!

## 💡 What is this?

This repo is part of my shift from pure mathematics into machine learning, with a specific focus on **interpretability** and **alignment**. While I have more advanced projects in progress (LLMs, AlphaZero-style agents, etc.), I wanted to start with something clean, visual, and controllable:

> A multi-layer perceptron trained on MNIST — the simplest neural microscope I could build.

I hope for this project to be both a **learning tool** and a **teaching resource**. It’s meant to walk through:
- How to build an MLP from scratch (mirroring my [micrograd](/zero-to-hero-course/episode-1) architecture)
- How to track internal activations and neuron behaviors
- How to begin asking interpretability questions about hidden representations

If you’re early in your ML journey, or just looking to see how models can be dissected from first principles — I hope this is useful.

## Goals (Live Roadmap)

- ✅ Build an MLP in PyTorch from scratch (mirroring micrograd)
- [ ] Train on MNIST and log internal states (activations, logits)
- [ ] Visualize training dynamics (loss curves, activation heatmaps)
- [ ] Apply PCA to intermediate activations
- [ ] Investigate neuron specialization and representation drift
- [ ] Extend to CNN for comparison
- [ ] Prepare model and logs for deeper interpretability tools (e.g., TransformerLens-style analysis)

## Notebooks (Growing List)

| Notebook | Purpose |
|---------|---------|
| `00_MLP_from_scratch_PyTorch.ipynb` | Scalar → MLP from first principles (for learning) |
| `01_MLP_for_Interpretability.ipynb` | Clean, inspectable PyTorch MLP with hooks |
| `02_Train_on_MNIST_and_Log.ipynb` | Full MNIST training loop with activation capture |
| `03_Analyze_Activations.ipynb` | PCA, neuron-level insights, visualizations |

## Philosophy

I learn by building — not just by reading. This repo reflects that. The goal isn’t to “beat MNIST,” but to **see what’s happening inside**. I want to understand how even small networks form internal representations, and how we might design tools to probe, interpret, and improve them.

If you have suggestions, corrections, or want to riff on any of this — open an issue or drop me a message.
  
