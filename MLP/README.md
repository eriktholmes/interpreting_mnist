# 🔍 MLP Exploration – Activation Dynamics & Normalization

This folder contains a deep dive into a simple Multilayer Perceptron (MLP) trained on MNIST, used as a controlled environment to investigate training dynamics, internal activations, and the effects of normalization. The goal is to build intuition and set the stage for more advanced interpretability tools.



## 📓 Notebook 01: Activation Dynamics in a Non-Normalized MLP (Part I)

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


### 📉 Summary & Takeaways

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
---


## 📓 Notebook 02: Activation Dynamics in a Non-Normalized MLP (Part II)
In this notebook we start where we left off with a basic MLP and train it without normalization. This time, we add hooks for both forward and backward analysis, tracking mean of both activations and gradients.
We focus here on individual neurons in the hidden layers, tracking activations as model trained to see how the neurons evolved. We see that certain neurons seemed to have increasing activations, while others flucutated, indicating that the model may have started to rely on a few key neurons for decision making: we visualize this via a heatmap of the mean neuron activation across epochs:
| Hidden layer 1| Hidden layer 2 |
| ---------| --------- |
|<img width="1391" alt="Screenshot 2025-07-04 at 12 29 25 PM" src="https://github.com/user-attachments/assets/69120700-b499-40c4-823a-9f5c89c12811" />|<img width="1376" alt="Screenshot 2025-07-04 at 12 29 14 PM" src="https://github.com/user-attachments/assets/2e3130fd-2ae1-4592-a88b-f137ac59f323" />|

which shows that neuron 2 and 9 (in the second hidden layer) were highly active at the end of training, which makes us suspect that these neurons are likely picking up on some key features in the data. To dig into this a bit more we ask what the top activating neurons are per class (hoping that this would give us information about the features that this is detecting). Using hooks we analyze this (per neuron) class sepcific behavior, we compute the gradients in the final linear layer (output) and activations in layer 2 to see what sort of overlap there might be: the hope here was to see which neurons are most impactful in terms of the loss and which have high activations and push the 
network towards a certain decision. We hoped the overlap might shed light on which neurons to look further into. 

|                      | Class 0 |  Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 | 
| -------------------- | ------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| Top mean activations | [6,2,8] | [0,9,10] | [9,13,6]| [9,6,12]| [2,8,5] |[6,10,12]| [8,2,10]|[2,12,14]| [2,9,10]| [2,9,12]|
| Top mean gradients   | [6,5,2] | [8,3,2]  | [3,8,0] | [5,8,2] | [9,6,8] | [3,8,0] | [5,4,0] | [9,2,3] | [5,3,6] | [4,7,8] | 

This turns out not to be as helpful as we hoped, but does indicate that neuron 2 and 9 are activate across multiple classes. This doesn't exactly help us undersand what features the neurons may be responding to or whether they are actually picking up on a specific feature at all. It is basically just showing us what we already knew, which was that neurons 2,6,and 9 and highly active across of the dataset. What we really hope to know is whether certain neurons favor certain classes, or at least favor certain classes in a way that helps us understand possible interpretable features. To dig into this we analyze the *class selectivity score*, which compares the mean activation of a single neuron for a fixed class to the mean activation of that neuron across ALL other classes: we wish to understand when a fixed neuron seems to be more active for on a certain class. This is the formula we use:

$$ S_{i}^{cl} = \frac{\mu_{i}^{cl} - (\mu_{i}^{cl})^c}{\mu_{i}^{cl} + (\mu_{i}^{cl})^c + \epsilon},$$

which yields the most 'selective' neurons per class:
|                          | Class 0 |  Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 | 
| --------------------     | ------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| Highly selective neurons | [6,8,12] | [0,3,1] |[15,13,9]| [4,6,5] | [5,2,8] | [3,7,6] |[8,13,10]|[14,4,12]| [7,0,1] | [14,5,7]|


This paints a bit better picture for us, showing that the highly active neurons (2, and 9) are not very 'class selective'. We do see that neuron 6, whcih was also active above, is highly selective for class 0, indicating that it may capture a feature of 0 that also appears in other classes. The other takeaways from this are that neuron 14 is the onnly neuron that appears as the top highly selective for two classes (7 and 9), Nueron 0 only appears in the list for classes 1 and 8 maybe signalling that it captures center strokes or symmetry in the data, and that neuron 15 only appears once in class 2. This last point warrents a bit more investigation because above it appeared almost dead in the training evolution above. We then investigate these 3 neurons further:

#### Neuron 14, first pass:
We first studied this using ablation: we compute the models accuracy on classes 7 and 9 as a baseline, then kill the neuron and compute the models accuracy again on these classes, and finally scale the activation of the neuron and again compute the models accuracy. We find:

|  Scaling factor  |  Class 7 Accuracy | Class 9 Accuracy |
| :--------------: | :---------------: | :--------------: | 
|       1.0        |    94.46%         |      94.46%      |
|       0.0        |       81.61%      |       53.32%     |
|       3.0        |      96.30%       |     97.22%       |


In fact, looking at all the classes, we see that as we scale the neurons activation we get an increase in logits for classes 7,9,2,5, and a decrease (or no affect) on the other classes. Finally, we consider a matrix which tracks the target (row) and the prediction (column), so a model that was 100% accurace would yield a diagonal matrix (or heatmap with only lights along the diagonal). Our model is NOT 100% accurate but we still have a decent baseline output:

<img width="30%" alt="Screenshot 2025-07-04 at 1 55 29 PM" src="https://github.com/user-attachments/assets/7af61dc5-490c-4229-ae07-bbe1a8325046" />

We infact will look at the difference of this baseline (ablation-free) matrix with the matrix after ablation to see exactly where the model is failing in its predictions. 
The result will be a heatmap where the intensity measures the amount of failure in a given row.  

#### A helper function ```analyze_neuron```
To analyze the other neurons we created a helper function that takes in the model, neuron index, dataset, scaling interval, and number of steps through the interval and returns the confusion matrix, and class based accuracy per scale.  We plot the results of ```analyze_neuron(model, neuron_idx, test_dataset, (0,3), 5)``` for each of the neurons mentioned above: 

|  Neuron 0   |  Neuron 7 | Neuron 14 | Neuron 15 | 
| :---------: | :-------: | :-------: | :-------: | 
|  <img width="861" alt="Screenshot 2025-07-04 at 1 42 50 PM" src="https://github.com/user-attachments/assets/1ffb7ae1-ee3f-442c-95e7-0e4a44f5f4af" />|<img width="858" alt="Screenshot 2025-07-04 at 2 02 29 PM" src="https://github.com/user-attachments/assets/a77aff2f-240d-4171-8459-b084d083deb2" />|<img width="848" alt="Screenshot 2025-07-04 at 1 43 17 PM" src="https://github.com/user-attachments/assets/3fad700a-c26a-4130-96fb-1cff6f60e39d" />|<img width="841" alt="Screenshot 2025-07-04 at 2 04 32 PM" src="https://github.com/user-attachments/assets/06333ef8-9f49-4c15-91ee-637e11405482" />|
|<img width="524" alt="Screenshot 2025-07-04 at 2 11 17 PM" src="https://github.com/user-attachments/assets/c728b9e2-1615-4b74-80d0-a6703bb9f2e2" />|<img width="521" alt="Screenshot 2025-07-04 at 2 12 04 PM" src="https://github.com/user-attachments/assets/cf03f4cb-2529-4304-8b91-89fc052f53d2" />|<img width="527" alt="Screenshot 2025-07-04 at 2 12 47 PM" src="https://github.com/user-attachments/assets/5d7cb0de-77af-4b88-b50e-0604df3af662" />|<img width="541" alt="Screenshot 2025-07-04 at 2 13 24 PM" src="https://github.com/user-attachments/assets/47a6c563-9591-42d0-9d71-0626e021bd13" />|













## 🚧 What's Next?
Upcoming notebooks will explore more interpretability tools with focus on some of the following (for now):

- Neuron death and saturated units
- Hooks for tracking activations and gradients
- PCA and class separation
- Normalization experiments for comparison
- Ultimately, this path builds toward interpretability in deeper and more expressive models like CNNs and Transformers.
