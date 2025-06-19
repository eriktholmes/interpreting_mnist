# 🔍 MLP Exploration – Activation Dynamics & Normalization

This folder contains a deep dive into a simple Multilayer Perceptron (MLP) trained on MNIST, used as a controlled environment to investigate training dynamics, internal activations, and the effects of normalization. The goal is to build intuition and set the stage for more advanced interpretability tools.

# 📂 Notebook Overview
## ```01_MLP_for_Interpretability_non_normalized.ipynb```

### Overview:
Train a minimal MLP on raw (non-normalized) MNIST data to explore activation behavior, drift, and statistical changes during training.

### Key Highlights:

- Manual tracking of layer activations over epochs
- Class-wise (digit based) activation visualizations
- We observe clear evidence of:
  - Activation drift (means changing over epochs)
  - Variance expansion (increased spread of activation values)
- Test accuracy reaches ~94%, yet internal stats show unstable representation dynamics. Here are some failed predictions:
  <img width="1207" alt="model_failures" src="https://github.com/user-attachments/assets/8e359bfa-4648-49c4-a8a4-a9326021f9cf" />


### Visualizations:
Here are a few of the visuals produced in the document. The first table highlights the spread (fanning out) of activations as the model trained. The second focuses specifically on digits (0 and 3) and shows how the mean and variance change over the course of training. 
|  Linear layer 1 activations across epochs  |   ReLU layer 1 activations across epochs     | 
| -------- | ------- |
| <img width="500" alt="Layer1_activations_across_epochs" src="https://github.com/user-attachments/assets/a0ac470d-ac03-4aa6-bd0d-2f0e7a8cf594" /> | <img width="500" alt="ReLU1_activations_across_epochs" src="https://github.com/user-attachments/assets/61eddf38-4671-4170-aef9-d1e48e2aa1df" /> |

| Class 0 mean/std evolution | Class 3 mean/std evolution |
| ---------| --------- |
|<img width="661" alt="class0_stats_across_epochs" src="https://github.com/user-attachments/assets/32acff13-0b6f-4bf5-a338-542cbcd7056e" />|<img width="625" alt="class3_stats_across_epochs" src="https://github.com/user-attachments/assets/bc2a2cbb-5fcb-4f12-b130-b6264529df13" />|





### Summary of findings:

- Early training caused a spike in activation mean-from ~0.3 to ~1.3 (depending on class)-followed by gradual decay
- Standard deviation increases steadily, indicating internal instability
- Model performs well externally but shows issues internally 

Ultimately, these results motivate the need for normalization techniques (e.g. BatchNorm, input scaling) to stabilize and improve representational health. That will follow in another notebook!


### 📌 Why Start Here?
This MLP serves as a clean, interpretable baseline.  Motivated by some of the Kaparthy Zero-to-Hero videos we figured MNIST was a nice place to start interpretability experiments and that we should really begin by investigating raw (non-normalized) data. 

---

## 🚧 What's Next?
Upcoming notebooks will explore more interpretability tools with focus on some of the following (for now):

- Neuron death and saturated units
- Hooks for tracking activations and gradients
- PCA and class separation
- Normalization experiments for comparison
- Ultimately, this path builds toward interpretability in deeper and more expressive models like CNNs and Transformers.
