# Robot-moving-using-Trust-region-policy-optimization-compared-to-proximal-policy-optimization


TRPO and PPO are algorithms used to optimize control policies in reinforcement learning. TRPO uses a trust region method to ensure conservative updates, while PPO uses proximal optimization and a clip objective for improved sample efficiency.   Both methods utilize the trust region method to increase reliability in finding the optimal policy for a robot's movement.

1   Initialize a new notebook in the Colab platform (or use a PC system with GPU) 

2   Install necessary libraries:
-gym
-pytorch
-visualization tools
-brax library (for environments)

3   Follow the file of implementation guide in the main repository files and follow the instructions carefully.




Implementation Guide: [Robot Control Using Trust Region Policy Optimization (TRPO) vs Proximal Policy Optimization (PPO)]:





installs the X Virtual Framebuffer (Xvfb) package:
to enables running graphical applications, such as gym, without the need for a physical display.

![image](https://user-images.githubusercontent.com/100143830/213734450-8f4b3b44-4682-4f22-85b3-d8c8d268d91b.png)


The following code installs the necessary dependencies for running a reinforcement learning algorithm on a virtual frame buffer:Its install gym==0.23.1, that provides a range of environments for developing and comparing reinforcement learning algorithms, pytorch-lightning==1.6:  its library that simplifies the prototyping and research process for deep learning models.
pyvirtualdisplay: A Python library that acts as a wrapper for Xvfb and other virtual display libraries.
 Install these packages together will allow you to run reinforcement learning algorithm on virtual frame buffer.










Install the brax library  from its Github repository. 
This library is a set of utilities and environments for reinforcement learning, so this package will make it easier to use and work with reinforcement learning environments and methods in your code.







Imports a variety of libraries that are commonly used in machine learning, reinforcement learning, and data visualization. Some of the specific functions and classes that are imported:



































The line of code device = 'cuda:0' is used to set the device to the first available GPU, with the index of 0, on which a tensor should be stored and operated on, It then gets the number of CUDA-enabled GPUs available on the system and assigns it to the num_gpus variable, then creates a 1-dimensional tensor of ones on the device specified in the device variable and assigns it to the v variable.






It's worth to mention that if you don't have GPU device on your system, this code will raise an error.









In this step uses the PyTorch library to create video function: create_video.
This function takes an environment, the number of steps the agent takes in the environment as input. The function uses the samples actions from the environment's action space, then it takes these actions in the environment and collects the states of the environment in an array. Finally, it returns a rendered video of the agent's actions in the environment, which allows the user to see how the agent is behaving in the environment.



















From PyTorch library create test_agent function to evaluate the performance of an agent in an environment. It takes an environment, the number of steps the agent takes in the environment, a policy function and the number of episodes as input. The function uses the policy to generate actions, then it takes these actions in the environment and accumulates the rewards. It repeats this process for a number of episodes, then it returns the average of the accumulated rewards as a performance metric of the agent. This function allows the user to evaluate the effectiveness of the agent's policy in the environment.

























This code defines a PyTorch neural network module called GradientPolicy. The network has two hidden layers with ReLU activations and two output layers, one for the mean values of the policy and the other for the standard deviation values. The mean values are passed through a tanh activation to limit the range while the standard deviation values are passed through a softplus activation and added with a small constant to ensure they are positive. The forward() method applies the linear layers and activations in sequence and outputs the mean and standard deviation tensors.




















Define the value network using PyTorch neural network module called ValueNet, with 2 hidden layers and 1 output layer, using ReLU activations. The network takes an input of size "in_features" and outputs a single value representing the predicted value of a given state or observation. The forward() method applies the linear layers and activations in sequence and outputs the predicted value. This network is commonly used as a critic in reinforcement learning tasks to estimate the value of a state or action.






























Create the RunningMeanStd class to keep track of the running mean and standard deviation of a stream of data. It is a way to calculate the mean and standard deviation of a large dataset, by processing the data in small segments and updating the running mean and standard deviation after each segment.








































Define the class "NormalizeObservation" is to normalize the observations coming from a gym environment by using the running mean and standard deviation. It wraps around a gym environment and normalizes the observations obtained from the environment before returning them.






















This code defines a function called create_env which takes three parameters env_name, num_envs and episode length, The function creates an instance of the gym environment by calling the gym.make() function with the given and the number of environments and the length of the episode as arguments. Then it wraps the environment with the "NormalizeObservation" class defined earlier. This class normalizes the observations coming from the environment by using the running mean and standard deviation, the function returns the wrapped environment. Then creates an environment for running the 'ant environment with a total of 10 parallel environments. The env.reset() function is then called, which resets the environment and returns the initial observation of the environment.







We have completed the main implementation steps of our project. The remaining details and procedures for execution can be found in the accompanying repository for reference, In order to implement the TRPO agent, we first implemented its optimizer and associated dataset. We then proceeded to implement the training code. Similarly, for the PPO agent, we first implemented the agent's data pipeline. Utilizing the TensorBoard tool, to visualize the results of both learned agents and compare the two results.




