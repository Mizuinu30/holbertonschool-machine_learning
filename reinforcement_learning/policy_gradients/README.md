
# **Policy Gradient Reinforcement Learning**

## **Description**
This project implements a Policy Gradient reinforcement learning algorithm using the Monte-Carlo policy gradient method (REINFORCE). It demonstrates how to compute policies, gradients, and train an agent in a Gymnasium environment. Additionally, it includes functionality for visualizing the training process.

---

## **Learning Objectives**
1. Understand what a policy is in reinforcement learning.
2. Learn how to calculate a policy gradient.
3. Implement and use the Monte-Carlo policy gradient (REINFORCE) algorithm.

---

## **Project Structure**
```
.
├── policy_gradient.py  # Contains functions for computing policies and gradients.
├── train.py            # Implements the training loop for the agent.
├── 0-main.py           # Example script for Task 0.
├── 1-main.py           # Example script for Task 1.
├── 2-main.py           # Example script for Task 2.
├── 3-main.py           # Example script for Task 3.
├── README.md           # Project documentation.
```

---

## **Requirements**
- Python 3.9
- Gymnasium (version 0.29.1)
- NumPy (version 1.25.2)
- Matplotlib (for visualization, optional)

Install dependencies with:
```bash
pip install numpy==1.25.2 gymnasium==0.29.1 matplotlib
```

---

## **Usage**
### **Task 0: Compute Policy**
Run:
```bash
python3 0-main.py
```
This calculates the policy for a given state and weight matrix using the softmax function.

---

### **Task 1: Compute Monte-Carlo Policy Gradient**
Run:
```bash
python3 1-main.py
```
This script computes the Monte-Carlo policy gradient for a given state and weight matrix. The output includes the chosen action and the computed gradient.

---

### **Task 2: Train the Agent**
Run:
```bash
python3 2-main.py
```
This script trains the agent in the `CartPole-v1` environment using the REINFORCE algorithm. The output includes the score of each episode, and a plot of the training scores will be displayed.

---

### **Task 3: Visualize Training**
Run:
```bash
python3 3-main.py
```
This script adds an option to visualize the training process. Rendering occurs every 1000 episodes.

---

## **Example Output**
### **Training Results (Task 2)**
The script outputs the score for each episode, e.g.:
```
Episode: 0 Score: 22.0
Episode: 1 Score: 62.0
...
Episode: 9999 Score: 500.0
```
A plot of scores over episodes is displayed at the end of the training.

---

## **References**
- [How Policy Gradient Reinforcement Learning Works](https://towardsdatascience.com/how-policy-gradient-reinforcement-learning-works-d2e0e3b99892)
- [Policy Gradients in a Nutshell](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#policy-gradient)
- [RL Course by David Silver - Lecture 7](https://www.davidsilver.uk/teaching/)
- [Policy Gradient Algorithms](https://towardsdatascience.com/policy-gradient-algorithms-5aecbaebc23f)

---

## **Author**
[Juan Silva]
Holberton School Machine Learning Track
November 2024

