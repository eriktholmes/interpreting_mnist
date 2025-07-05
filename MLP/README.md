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






---




## 📓 Notebook 02: Activation Dynamics in a Non-Normalized MLP (Part II)
In this notebook we extend our analysis of a basic 2-layer MLP trained without normalization. The goal is to build up our tools and intuition around individual neurons and the layers that they reside within. Towards this understanding we utilize forward (activation) and backward (gradient) hooks to gather data (per-neuron) which we then dig into as the notebook proceeds. 


### Part 1: Training evolution
We focus first on individual neurons in the hidden layers, tracking activations as the model trains to see how these activations evolve and on which neurons the model starts to rely. Below are two heatmaps of mean activations per neuron for both of the hidden layers:
| Hidden layer 1| Hidden layer 2 |
| ---------| --------- |
|<img width="1391" alt="Screenshot 2025-07-04 at 12 29 25 PM" src="https://github.com/user-attachments/assets/69120700-b499-40c4-823a-9f5c89c12811" />|<img width="1376" alt="Screenshot 2025-07-04 at 12 29 14 PM" src="https://github.com/user-attachments/assets/2e3130fd-2ae1-4592-a88b-f137ac59f323" />|

We see that neurons 2,6, and 9 in Layer 2 become highly active in the training process and suspect that they play a large role in the models decision making. 

### Part 2: per-neuron activations and gradient

To dig into this a bit more we ask what the top activating neurons are per class (hoping that this would give us information about the features that this is detecting). For each class, and across all inputs of that class, we compute the top 3 neurons by both mean activation and mean gradient. The hope was that overlap between these two measures would point us in the direction of neurons that were both active and important to the models predictions. 



|                      | Class 0 |  Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 | 
| -------------------- | ------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| Top mean activations | [6,2,8] | [0,9,10] | [9,13,6]| [9,6,12]| [2,8,5] |[6,10,12]| [8,2,10]|[2,12,14]| [2,9,10]| [2,9,12]|
| Top mean gradients   | [10,9,7]| [2,7,6]  | [7,1,5] | [0,2,10]| [9,6,14]| [1,2,0] |[12,5,14]| [7,10,8]| [5,4,8] |[13,10,4]| 

This table shows that certain neurons — notably 2, 6, and 9 — are consistently active across classes, but they don’t exhibit strong class specificity. These appear to function as generalist neurons, contributing to multiple class logits rather than specializing in one. While this aligns with our intuition from earlier training observations, it left open the central question:
*Do any neurons consistently favor certain classes in a way that reveals interpretable structure?*


> Upon reflection (during the writing of this) I realized that computing the gradients with respect to the loss (which involves all the logits and is reflective of more global model behavior) was not the correct approach to answer this question and was a bit more confusing than it needed to be. I should instead try to understand how influencial each neuron is to the class the model is trying to predict, $y$ (i.e. understanding more local behavior of the model).


In code, we address the by hooking the gradients after calling ```logits[y].backward``` rather than ```loss.backward()```. Doing this we obtain a slightly more informative table where overlap between columnns indicates neurons that are both very active, AND contributing to the correct prediction, which is hopefully more interpretable!

|                      | Class 0 |  Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 | 
| -------------------- | ------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| Top mean activations | [6,2,8] | [0,9,10] | [9,13,6]| [9,6,12]| [2,8,5] |[6,10,12]| [8,2,10]|[2,12,14]| [2,9,10]| [2,9,12]|
| Top mean gradients   |[6,12,13]| [0,1,9]  | [13,9,2]| [9,4,5] | [5,2,8] |[10,12,7]| [8,7,13]| [4,2,14]| [7,1,10]| [7,14,9]| 



### Part 3: Class selectitivity
Towards understanding neurons that are fire more for certain classes we compute the **class selectivity score** for each neuron $i$ and class $cl$.  

$$ S_{i}^{cl} = \frac{\mu_{i}^{cl} - (\mu_{i}^{rest})}{\mu_{i}^{cl} + (\mu_{i}^{rest}) + \epsilon},$$

This tells us whether the ith neuron is more active for class cl than for all other classes combined, and we obtain the following table:


|                          | Class 0 |  Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 | 
| --------------------     | ------- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | 
| Highly selective neurons | [6,8,12] | [0,3,1] |[15,13,9]| [4,6,5] | [5,2,8] | [3,7,6] |[8,13,10]|[14,4,12]| [7,0,1] | [14,5,7]|


This paints a bit better picture for us, showing us neurons that are more class specific (or *class specialists*). Here are some quick observations:
- neuron 6, which we deemed may be a generalist neuron, is highly selective for class 0, indicating that it may capture a feature of 0 that also appears in other classes.
- Neuron 14 is selective for two classes (7 and 9) indicating that is may be polysemantic.
- Nueron 0 only appears in the list for classes 1 and 8 maybe signalling that it captures center strokes or symmetry in the data
- Neuron 15 is selective for class 2 despite seeming nearly dead in the training plots above.
There are more observations to be made (fun for the reader!) but we dive into these ones below. 



### Part 4.0: Scaling and Ablation (Neuron 14)

This was a curious case where neuron 14 was selective for two classes so we started here. We compute the models accuracy on classes 7 and 9 as a baseline, then zero out the neurons activations (ablation) and compute the models accuracy again on these classes, and finally scale the activation of the neuron and again compute the models accuracy. We find:

|  Scaling factor  |  Class 7 Accuracy | Class 9 Accuracy |
| :--------------: | :---------------: | :--------------: | 
| 1.0 (baseline)   |    94.46%         |      94.46%      |
|  0.0 (ablation)  |       81.61%      |       53.32%     |
|  3.0 (scaled)    |      96.30%       |     97.22%       |

In fact, looking at all the classes, we see that as we scale the neurons activation we get an increase in logits for classes 7,9,2,5, and a decrease (or no affect) on the other classes. To see what is going on with the model under these scalings we consider a matrix whose (i,j) entry counts the number of times that a class i input is classified as class j. We call this the *confusion matrix* and not that if a model is 100% accuracte we would have a diagonal matrix (no confusion takes place). Our model is NOT 100% accurate but we still have a decent (baseline) output when no neurons are tampered with:

<img width="30%" alt="Screenshot 2025-07-04 at 1 55 29 PM" src="https://github.com/user-attachments/assets/7af61dc5-490c-4229-ae07-bbe1a8325046" />

To get a better idea of what is going 'wrong' in the model we will look at the difference of the baseline confusion matrix and the ablation confusion matrix. We show this result in the next section. 


### Part 4.1: Generalizing this with a helper function ```analyze_neuron```
We created a helper function that takes in the model (speific to our setup for now), neuron index, dataset, scaling interval, and number of steps through the interval and returns a plot of class based accuracy under scaling, as well as the confusion matrix: under ablation, baseline, and under 'max' scaling, where max is just the right endpoint of the scaling interval.  We attempt to use these visualizations to 

We plot the results of ```analyze_neuron(model, neuron_idx, test_dataset, (0,3), 5)``` for each of the neurons mentioned above: 

|     |  Neuron 0   |  Neuron 6 | Neuron 7 | Neuron 14 | Neuron 15 | 
| --- | :---------: | :-------: |:-------: | :-------: | :-------: | 
|Class based accuracy|  <img width="861" alt="Screenshot 2025-07-04 at 1 42 50 PM" src="https://github.com/user-attachments/assets/1ffb7ae1-ee3f-442c-95e7-0e4a44f5f4af" />|<img width="871" alt="Screenshot 2025-07-05 at 5 46 51 PM" src="https://github.com/user-attachments/assets/71019267-ea5b-4088-98ad-3695376ad478" />|<img width="851" alt="Screenshot 2025-07-05 at 3 15 01 PM" src="https://github.com/user-attachments/assets/22d1c420-0d59-4ef0-87d3-0274d39b6ca3" />|<img width="863" alt="Screenshot 2025-07-05 at 3 17 06 PM" src="https://github.com/user-attachments/assets/43e2015a-ebde-47be-8e05-cf8a1f1ce997" />|<img width="841" alt="Screenshot 2025-07-05 at 3 18 36 PM" src="https://github.com/user-attachments/assets/f0d7c8e2-0574-43c7-b3fb-a34c97fd25fa" />|
|baseline-ablation matrix|<img width="524" alt="Screenshot 2025-07-04 at 2 11 17 PM" src="https://github.com/user-attachments/assets/c728b9e2-1615-4b74-80d0-a6703bb9f2e2" />|<img width="533" alt="Screenshot 2025-07-05 at 5 47 08 PM" src="https://github.com/user-attachments/assets/d1adf7cf-fcd5-452d-87cb-c3a904f3fc63" />|<img width="521" alt="Screenshot 2025-07-04 at 2 12 04 PM" src="https://github.com/user-attachments/assets/cf03f4cb-2529-4304-8b91-89fc052f53d2" />|<img width="523" alt="Screenshot 2025-07-05 at 3 19 58 PM" src="https://github.com/user-attachments/assets/9e58099f-8f94-4198-a005-5b9c8e609859" />|<img width="541" alt="Screenshot 2025-07-04 at 2 13 24 PM" src="https://github.com/user-attachments/assets/47a6c563-9591-42d0-9d71-0626e021bd13" />|
|baseline-max_scaled matrix|<img width="524" alt="Screenshot 2025-07-05 at 5 32 07 PM" src="https://github.com/user-attachments/assets/b0284b67-261d-41c6-a480-9d9082708325" />|<img width="527" alt="Screenshot 2025-07-05 at 5 47 17 PM" src="https://github.com/user-attachments/assets/d7611ff6-ef44-4b46-a634-ba5afe1fc3d4" />|<img width="520" alt="Screenshot 2025-07-05 at 5 32 51 PM" src="https://github.com/user-attachments/assets/5b47fa5b-123e-4143-a4f2-3416f02b5a74" />|<img width="527" alt="Screenshot 2025-07-05 at 5 33 36 PM" src="https://github.com/user-attachments/assets/d124b9fe-b8c8-46c4-9b76-940fa992511a" />|<img width="516" alt="Screenshot 2025-07-05 at 5 48 49 PM" src="https://github.com/user-attachments/assets/ccd1d2c0-bfec-4b0d-a2b6-649dfc07105e" />|










We see some interesting behavior and summarize our results here:


> **Neuron 0**:
>  - Highly selective for class 1
>  - Under ablation the accuracy of the model drops to 0%
>  - Scaling increases class 1 accuracy monotonically
>  - The confusion matrix identifies that this neurons ablation causes confusion between 1 and 3
>
> This shows that, while the model is quite good at prediciting the class 1, this predicition lies on the shoulders of neuron 0!

> **Neurons 6 (and 7)**: we observe similar behavior from these two neurons so summarize the results here
>  - Highly selective for class 0 (classes 8 and 9)
>  - Under ablation the class 0 (class 9) accuracy of the model drops to <20% (<25%)
>  - Scaling increases class 0 (classes 8 and 9) accuracy monotonically
>  - Acts as an attractor under scaling, causing most digits to be classified as 0 (8)
> 
> While scaling increasese accuracy of the class 0 (classes 8 and 9) we find degradation in other classes. It likely activates for some (shared) curvature feature(s). 



> **Neuron 14**:
>  - Highly selective for classes 7 and 9
>  - Under ablation the accuracy of the model drops below 50% for class 9 but only to around 85% for class 7.
>  - Scaling increases classes 7 and 9 accuracy monotonically but decreases all other classes. 
>  - The confusion matrix identifies that under ablation the model misclassifies 9 as 4
>  - Under scaling we see the primary misclassification is the opposite, 4 is misclassified as 9
>
> This neuron seems somewhat unique in that the confusion is somehow bi-directional, and affects classes 4 and 9. This is interesting, as it is also very selective for class 7, it seems likely that the feature involves both a vertical stroke and maybe a top horizontal stroke?



> **Neuron 15**:
>  - To be filled in soon... 










## 🚧 What's Next?
Upcoming notebooks will explore more interpretability tools with focus on some of the following (for now):

- Neuron death and saturated units
- Hooks for tracking activations and gradients
- PCA and class separation
- Normalization experiments for comparison
- Ultimately, this path builds toward interpretability in deeper and more expressive models like CNNs and Transformers.
