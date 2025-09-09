
# Reinforcement Learning (RL) Paper Domains

*I am working on constructing a general RL paper repository. Currently I am reading RL related paper in details, and very happy to invite researchers who are interested in building this repository together with me. Most of my resources are from Google Scholar, Openreview, ACM Digital Libararies, Top-Tier AI Conferences, and open-source github repositories such as OpenDILab.*

## Classical methods 

* [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (DQN)
* [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) (DRQN)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (Dueling-DQN)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (Double-DQN)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (PEM)
* [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) (Rainbow)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (TRPO)
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (A3C)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (GAE)
* [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224) (ACER)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO)
* [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144) (K-FAC-TPO)
* [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (SAC)
* [Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.pdf) (DPG)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (DDPG)
* [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (TD3)
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) (C51))
* [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) (QR-DQN)
* [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923) (IQN)

## Model-based RL 

### Representative Paper

* [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://dl.acm.org/doi/10.5555/3104482.3104541) (Gaussian Process-Based Probabilistic Modelling)
* [Learning Complex Neural Network Policies with Trajectory Optimization](https://proceedings.mlr.press/v32/levine14.html) (Iterative Linear Quadratic Gaussian-Based Trejctory Optimization and Policy Learning)
* [Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142)
* [Value Prediction Network](https://arxiv.org/abs/1707.03497)
* [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675)
* [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999)
* [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)
* [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
* [Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/abs/1807.03858)
* [Model-Ensemble Trust-Region Policy Optimization](https://openreview.net/forum?id=SJJinbWRZ)
* [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
* [Exploring Model-based Planning with Policy Networks](https://openreview.net/forum?id=H1exf64KwH)
* [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)
* [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)

### Survey and Review

* [Model-based Reinforcement Learning: A Survey](https://ieeexplore.ieee.org/document/10007800)
* [Model-Based or Model-Free, a Review of Approaches in Reinforcement Learning](https://ieeexplore.ieee.org/document/9275964)
* [A Survey on Model-based Reinforcement Learning](https://github.com/lizhuo-1994/RL_paper/blob/main/s11432-022-3696-5.pdf)



## Offline RL 

### Representative Paper
* [AlgaeDICE: Policy Gradient from Arbitrary Experience](https://arxiv.org/pdf/1912.02074)
* [Behavior Regularized Offline Reinforcement Learning](https://arxiv.org/pdf/1911.11361)
* [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://xbpeng.github.io/projects/AWR/AWR_2019.pdf)
* [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf)
* [MOReL: Model-Based Offline Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/f7efa4f864ae9b88d43527f4b14f750f-Paper.pdf)
* [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900)
* [An Optimistic Perspective on Offline Reinforcement Learning](https://arxiv.org/pdf/1907.04543)
* [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/pdf/2110.06169)
* [A Closer Look at Offline RL Agents](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3908cadfcc99db12001eafb1207353e9-Abstract-Conference.html)
* [Conservative state value estimation for offline reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html)
* [State-Action Similarity-Based Representations for Off-Policy Evaluation](https://arxiv.org/pdf/2310.18409)
* [Comparing Model-free and Model-based Algorithms for Offline Reinforcement Learning](https://arxiv.org/pdf/2201.05433)
* [Efficient Online Reinforcement Learning with Offline Data](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)
* [The Generalization Gap in Offline Reinforcement Learning](https://openreview.net/pdf?id=3w6xuXDOdY)
* [A2PO: Towards Effective Offline Reinforcement Learning from an Advantage-aware Perspective](https://proceedings.neurips.cc/paper_files/paper/2024/file/333a7697dbb67f09249337f81c27d749-Paper-Conference.pdf)
* [Offline Reinforcement Learning with OOD State Correction and OOD Action Suppression](https://proceedings.neurips.cc/paper_files/paper/2024/file/a9f3457fa97f106f1756885237787789-Paper-Conference.pdf)
* [Enhancing Value Function Estimation through First-Order State-Action Dynamics in Offline Reinforcement Learning](https://proceedings.mlr.press/v235/lien24a.html)
* [Adaptive Advantage-Guided Policy Regularization for Offline Reinforcement Learning](https://openreview.net/pdf?id=FV3kY9FBW6)
* [Q-value Regularized Transformer for Offline Reinforcement Learning](https://openreview.net/pdf?id=ojtddicekd)
* [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://openreview.net/pdf?id=AHvFDPi-FA)
* [Adversarial Model for Offline Reinforcement Learning](https://arxiv.org/pdf/2302.11048)

### Survey and Review
* [Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2307.15217)
* [A Survey on Offline Model-Based Reinforcement Learning](https://arxiv.org/abs/2305.03360)
* [A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://arxiv.org/abs/2203.01387)
* [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)

## Multi-Agent RL 
## Hierarchical RL 
## Meta-RL 
## Memory-Based RL
## Exploration in RL
## Safe RL 
## Goal-Conditioned RL
## Explainable RL
## Adversial RL
## Imitation RL
## Privacy in RL
## Inverse RL

### Representative Paper


### Survey and Review
* [A survey of inverse reinforcement learning](https://link.springer.com/article/10.1007/s10462-021-10108-x)


## Transformer in RL
## Causal RL
## RL for LLM
## RL for Multi-Modal
## RL for Robotics

### Representative Paper
* [Large Language Models as Generalizable Policies for Embodied Tasks](https://arxiv.org/pdf/2310.17722)

### Survey and Review

## RL for Autonomous Driving Systems
## RL for Generative Models
## RL for Finance
## RL for Healthcare
## RL for Transportation
## RL Benchmarks
## RL for Game


