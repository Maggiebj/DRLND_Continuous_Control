## Report
### Algorithm Introduction
In this project, we applied Actor-Critic model-free algorithm, base on the paper [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971). The algorithm adapts the ideas underlying the success of Deep Q-Learning to the continous action domain. The algorithm is called DDPG(Deep Deterministic Policy Gradient).
DDPG contains two deep neural network: Actor and Critic. The Actor is a policy network which takes the states as inputs and output an action (in continuous space) which has highest Q state-action value. The Critic is a Q value network which takes state and action as input and output Q value. 
As Deep Q network, DDPG is an off policy network. Off policy learning is when the policy used for interacting with the environment(action) is different than the policy being learned. DDPG is deterministic since Actor output the best believed action for the given state rather than a probability distribution over actions.
To resolve a naive application of Actor-Critic method with neural function approximators having unstable problems, we have applied these methods:
1. replay buffer to minimize correlations between samples
2. each network has separate local network and target network, target network is to give consistent prediction while local network is to train and learn.
3. soft update. We update target network using TAU which is very small to change little by little from local network to improve stability of learning.
   $$\theta_{target} = \tau\theta_{local}  + (1-\tau)\theta_{target}$$
5. batch normalization. We using batch normalization to scale the features so thay are in similar ranges across environments and units. The technique normalize each dimension across the samples in a minibatch to have unit mean and variance.
6. update interval and learning interval. Skipping some timesteps to learn the local network and every 2 learnsteps update the target network
7. adding noise to action. We constructed an exploration policy by adding noise sampled from a noise process to our actor policy. Pay attention to reset the noise process after each adding.
   $$x_{noise} = x + \theta (\mu - x) + \sigma Unif(a)$$
where
- $x$ is the original state value
- $x_{noise}$ is the state value with noise
- $\mu$ is 0
- $\theta$ is 0.15
- $\sigma$ is 0.2
- $Unif(a)$ returns an array of random numbers with the action size and following a Uniform Distribution.

### Implementation

**Highlevel pseudo-code of the algorithm**
   Init random weights for  target and local critic and actor.
   Init replay memory
   Init noise process(OU noise)
   foreach episode
        Initialize a random process for action exploration (OU noise)
        Get initial state
        for each step in episode:
            Choose an action using actor policy and exploration noise
            Take action and observe the reward and next state
            Store the experience tuple in the replay buffer
    
            if we have enough experiences(batchsize):
                Get a batchsize of tuples 
                Get predicted next-state actions and Q values from target models
                Compute Q targets for current states (y_i)
                Compute Q expected from local model(Q_expected)
                Compute critic loss(y_i and Q_expected)
                Minimize the critic loss
                Compute actor loss
                Minimize the actor loss
                
                if every 2 learning steps:
                    soft update critic and actor target
                    
                
   **Hyperparameters**

        BUFFER_SIZE = int(1e6)  # replay buffer size
        BATCH_SIZE = 64         # minibatch size
        GAMMA = 0.95            # discount factor
        TAU = 1e-3              # for soft update of target parameters
        LR_ACTOR = 1e-3         # learning rate of the actor 
        LR_CRITIC = 1e-3        # learning rate of the critic
        WEIGHT_DECAY = 0        # L2 weight decay
        UPDATE_EVERY = 10       # skip learning step
    
    **Neural Networks**

    We have actor network, critic network and their target clones.
     
    The architecture is multi-layer perceptron. Input layer matches the state size then we have 3 hidden fully connected layers ,each layer followed by a batchnorm layer and finaly the output layer.
    
    Critics:
        state_size(33),,action_size(4) * 256 * 256 * 64 --> output 1 value (The Q value of the action,state pair)
        
    Actors:
        state_size(33) * 256 * 256 * 64 --> output action_size values(size 4) (the best believed action to perform)



### Code Structure

- ddpg_model.py contains the Torch implementation of the Actor Critic neural networks that are being used in the project.

- ddpg_agent.py contains the core of the project, such as agent.learn(),agent.step(),agent.act() and noise process, replay buffer

- Continuous_Control.ipynb puts togheter all the pieces to build and train the agent.

### Result
After around 500 episodes the avg score_deque(20) higher than 30
[scores on episodes](score.png)

### Ideas for Future Work

To improve the agent's performance we can also try DQN related improvements like prioritized experienced replay. We can also try more complicated network like more layers and weights.
