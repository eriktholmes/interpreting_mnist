# 🔍 MLP Exploration – Activation Dynamics & Normalization

This folder contains a deep dive into a simple Multilayer Perceptron (MLP) trained on MNIST, used as a controlled environment to investigate training dynamics, internal activations, and the effects of normalization. The goal is to build intuition and set the stage for more advanced interpretability tools.



# 📘 Notebook 01: Activation Dynamics in a Non-Normalized MLP

This notebook analyzes the internal behavior of a simple MLP trained on raw (non-normalized) MNIST data. Our focus is on **tracking representational drift**—how layer activations evolve across epochs—and identifying early signs of **instability or inefficiency** in learning.

The aim is to establish baseline behavior before introducing normalization or architectural changes.

### Key Experiments & Observations

- Tracked pre- and post-activation distributions across training epochs.
- Visualized both global and class-based activation histograms.
- Quantified activation drift by computing layerwise means and standard deviations.

**Observations:**
- *Mean activation drift*: All layers exhibit non-trivial shift in mean activation during early epochs, followed by partial stabilization.
- *Variance expansion*: Standard deviation increases consistently, suggesting uncontrolled growth in representation scale.
- Despite ~94% test accuracy, internal signals suggest inefficient or unstable learning dynamics.




### Visualizations:

We first highlight a few misclassified examples:
 <img width="1207" alt="model_failures" src="https://github.com/user-attachments/assets/8e359bfa-4648-49c4-a8a4-a9326021f9cf" />
> This prompted a quick test regarding class-based accuracy: we found (as suggested by the figure) that the model underperforms on digits like 2, 5, 8, and 9 which share overlapping visual features (e.g., loops, curvature). This motivates a *neuron-level* analysis which we will investigate more in the next notebook. 

Next, we visualize activation statistics: the first pair of plots show how activation distributions in the first linear and ReLU layers evolve across epochs. Note the **broadening of the histograms**—an early sign of representational drift.
|  Linear layer 1 activations across epochs  |   ReLU layer 1 activations across epochs     | 
| -------- | ------- |
| <img width="500" alt="Layer1_activations_across_epochs" src="https://github.com/user-attachments/assets/a0ac470d-ac03-4aa6-bd0d-2f0e7a8cf594" /> | <img width="500" alt="ReLU1_activations_across_epochs" src="https://github.com/user-attachments/assets/61eddf38-4671-4170-aef9-d1e48e2aa1df" /> |

This second set focuses on the digits **0** and **8**, chosen for their visual symmetry and since the model performs much better on the class 0 than 8. Mean spikes early; variance grows steadily.

| Class 0 mean/std evolution | Class 8 mean/std evolution |
| ---------| --------- |
|<img width="661" alt="class0_stats_across_epochs" src="https://github.com/user-attachments/assets/32acff13-0b6f-4bf5-a338-542cbcd7056e" />|<img width="625" alt="class3_stats_across_epochs" src="https://github.com/user-attachments/assets/bc2a2cbb-5fcb-4f12-b130-b6264529df13" />|


## 📉 Summary & Takeaways

- **Activation means** drift across epochs: experiencing sharp spikes across classes early in training, with gradual decline but remain elevated. 
- **Variance increases** across epochs—potential early sign of exploding activations.
- **Accuracy ≠ stability**: Despite good test performance on this shallow network, our experiments raise concerns for scalability. We also noted above the class based accuracy which we show here for readability:
    | Digit | Accuracy |   | Digit | Accuracy |
    |-------|----------|---|--------|----------|
    |   0   |  98.27%  |   |   1    |  98.15%  |
    |   2   |  92.93%  |   |   3    |  94.65%  |
    |   4   |  95.32%  |   |   5    |  92.60%  |
    |   6   |  95.93%  |   |   7    |  93.97%  |
    |   8   |  91.17%  |   |   9    |  91.08%  |

These findings motivate a follow-up exploration of neuron health and specialization and gradient analysis-covered in **Notebook 02**. 




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
