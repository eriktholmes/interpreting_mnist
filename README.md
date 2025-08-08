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
This repo explores the internal workings of image classifiers - to break apart/intervene in the black box and try to interpret the inner workings. The goal is to build and document real interpretability tools — starting from first principles and scaling up.

The project is structured as two parallel tracks (for now): one using a basic MLP, the other (in progress) extending to a CNN. Each track is presented through annotated notebooks with exploratory commentary, code, and visualizations. The final goal is to distill these findings into a blog post/write-up.

### Part 1: [`MLP`](https://github.com/eriktholmes/interpreting_mnist/tree/main/MLP):
We dedicate three notebooks to this experiment: throughout our model is a 2 layer MLP (32 -> 16 neuron hidden layers). We use SGD and fixed learning rate over 20 epochs.

All analysis is on MNIST. The track is organized into three notebooks:


- **Notebook 1: (unnormalized baseline)**
  >
  > Some experiments include:
  > - track neuron activations across epochs: manually in this notebook, then with hooks in notebook 2.
  > - Class-based activation statistics to identify class-specific or feature-detecting neurons
  > - Activation drift analysis, showing how selectivity changes over time
  > - Dimensionality reduction (PCA, t-SNE, UMAP) applied to the final hidden layer
 
- **Notebook 2**: (further unnormalized investigations)
  > We go deeper into interpretability and causality:
  > - Compute top-activating neurons per class — many show cross-class activation (shared features)
  > - Introduce selectivity scores, comparing mean activation on a class vs. others
  > - Perform causal interventions:
  >     - Neuron ablation: zeroing out individual neurons
  >     - Neuron scaling: multiplying activation values
  > - Apply dimensionality reduction techniques on highly selective subspaces:
  >     - take the top-$k$ neurons that are highly selective on a fixed class and apply UMAP on the $k$-dimensional subspace spanned by these neurons.
  > We see:
  > - Bottleneck neurons that control accuracy for specific classes — ablating them drops class accuracy to ~0%; scaling increases it monotonically
  > - A confusion matrix showing clear misclassifications between structurally similar digits (e.g., 4, 7, 9)
  > - Some class clustering (UMAP) on selective subspaces but its fairly noisy.


- **Notebook 3**: with normalization and dropout (currently being cleaned up)
  > We repeat the same experiment setup with:
  > - Normalized inputs
  > - Layer normalization
  > - Dropout (to prevent overfitting)
  > We observe:
  > - Faster training: matches baseline accuracy in a quarter to half of the epochs
  > - Better generalization: reduced overfitting and more stable neuron behavior
  > - UMAP on top selective neurons shows clearer clustering with possible submanifold shadows — the data forms curves or paths in the latent space, suggesting underlying geometric structure --> We are investigating this further:


- **Some visualizations:**
  > |     |   Activation drift   |  Neuron scaling | Confusion Matrix | UMAP class selective subspace |
  > | --- | :---------: | :-------: |:-------: | :-------: |
  > |Non-normalized| <img width="736" height="570" alt="Screenshot 2025-07-30 at 3 16 32 PM" src="https://github.com/user-attachments/assets/b35bbe13-b30d-4d74-8ebc-29d745cad1d0" />|<img width="871" alt="Screenshot 2025-07-05 at 5 46 51 PM" src="https://github.com/user-attachments/assets/71019267-ea5b-4088-98ad-3695376ad478" />|<img width="521" alt="Screenshot 2025-07-04 at 2 12 04 PM" src="https://github.com/user-attachments/assets/cf03f4cb-2529-4304-8b91-89fc052f53d2" />|<img width="927" height="671" alt="Screenshot 2025-08-06 at 11 46 30 AM" src="https://github.com/user-attachments/assets/cd13d09b-87d8-4882-afa6-75881d6be62d" />|
  > |Normalized|<img width="743" height="562" alt="Screenshot 2025-08-06 at 11 56 54 AM" src="https://github.com/user-attachments/assets/d4d00adb-16fb-4640-b0e7-b72382ed8dda" />|<img width="869" height="660" alt="Screenshot 2025-08-06 at 11 39 05 AM" src="https://github.com/user-attachments/assets/8b857c5d-a2e7-4ab3-9fdb-d60a60bee24d" />|<img width="535" height="544" alt="Screenshot 2025-08-06 at 11 39 24 AM" src="https://github.com/user-attachments/assets/7a857f8f-58a5-413f-9ef3-1f11343a86da" />|<img width="883" height="674" alt="Screenshot 2025-08-06 at 11 41 52 AM" src="https://github.com/user-attachments/assets/98bc797e-9976-4de1-ac0c-e09687cf8103" />|


-  I have numerous questions/experiments in mind at this point: from selectively based prune and its affect on model interpretability, to selectively based subspace analysis, and ultimately to compare the MLP experiments with CNN vs Transformer architechures. In any case, there will be more to come!


### Part 2: `CNN`
- Coming Soon!



<br> 

## Notebooks (Growing List)

| Notebook | Purpose |
|---------|---------|
| `01_MLP_for_Interpretability_non_normalized.ipynb` | Interpretable PyTorch MLP trained on raw MNIST input (hooks, dead neurons, histograms) |
| `02_MLP_for_Interpretability_non_normalized.ipynb` | Normalized-input training + PCA/activation comparisons |
| `03_MLP_for_Interpretability_normalized.ipynb` | 
| `03_CNN_for_Interpretability.ipynb`| (planned) Convolutional analog of the above, with same diagnostics|



## 🤔 Why MNIST?

MNIST is simple enough to fully analyze, yet rich enough to showcase key ideas and build/test mechanisms towards interpretability.

This project reflects how I learn best: by building, visualizing, and asking questions. It’s a first step in a larger interpretability journey — and a foundation for future experiments in Transformer internals and LLM behavior.

If you have suggestions, corrections, or other feedback let me know!
  
