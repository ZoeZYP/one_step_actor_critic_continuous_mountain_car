import numpy as np
import time
import matplotlib.pyplot as plt

import gym

"""
Simple implementation of one step actor-critic method in mountain car environment using continuous actions.
"""

#Simulation parameters
NUM_EPISODES = 10000
MAX_T = 200
ALPHA_W = 0.05
ALPHA_THETA_MU = 0.1
SIGMA_START = 0.1
SIGMA_FINAL = 0.001
SIGMA = 0
GAMMA = 0.99

#Test flgas
DEBUG = False
RENDER_POLICY = True
NUM_EPISODES_PLOT = 1000
SIGMA_DECAY = False

def linear_decay(current_time, final_time, initial_value, final_value):
    """
    It decrease linearly a value based on the current time.
    """
    return final_value + (initial_value - initial_value * (current_time/final_time))

def get_state(env):
    """
    It calculates the vector representation of the current state. In this case,
    it is used a state aggregation representation.
    """
    segmentation_factor = 100 # number of partition on each feature
    pos_segment = (env.high_state[0] - env.low_state[0]) / segmentation_factor
    vel_segment = (env.high_state[1] - env.low_state[1]) / segmentation_factor
    state = env.state
    coarse_state = np.zeros(2*segmentation_factor)

    coarse_state[int((state[0] - env.low_state[0])/ pos_segment)] = 1

    coarse_state[int((state[1] - env.low_state[1])/ vel_segment) + segmentation_factor] = 1

    return coarse_state

def value_approx(state, weights):
    """
    It calculates the value of the state-action pair multiplying the state-action
    pair by the learned weights.
    """
    return np.dot(state, weights)

def value_approx_grad(state, weights):
    """
    It calculates the value of the state-action pair multiplying the state-action
    pair by the learned weights.
    NOTE: In this case (linear case), this function is useless. Modify it if you
    change the value_approx fuction.
    """
    return state

def compute_mu(s,theta_mu):
    return np.tanh(np.dot(s,theta_mu))

def policy(env,s,theta):
    """
    It calculates the policy function and return the probability of executing
    each action. In this case it is used a gaussian policy.
    """
    mu = compute_mu(s,theta)
    return np.random.normal(mu, SIGMA)

def policy_grad(env,a,s,theta):
    """
    It calculates the gradient of the gaussian policy.
    """
    mu = np.tanh(s * theta)
    theta_mu_grad = (1 / SIGMA ** 2) * (a - mu) * s

    return theta_mu_grad


def training(env, w, theta, rewards):
    """
    It executes NUM_EPISODES episodes of training and returns the total rewards
    and the weights learned (weights w for the value approximation function and
    theta for the policy function).
    """
    for episode in range(NUM_EPISODES):
        time_start = time.time()
        env.reset()
        total_reward = 0

        for t in range(MAX_T):
            s = get_state(env)
            a = policy(env,s,theta)
            _, reward, done, _ = env.step([a])
            s_next = get_state(env)
            delta = reward + value_approx(s_next,w) - value_approx(s,w)
            w = w + ALPHA_W * delta * value_approx_grad(s,w)
            theta = theta + ALPHA_THETA_MU * delta * policy_grad(env,a,s,theta)

            s = s_next
            total_reward = total_reward + reward

            if done:
                break

        rewards[episode] = total_reward
        print("episode time: ", time.time()-time_start)
        if SIGMA_DECAY:
            SIGMA = linear_decay(episode, NUM_EPISODES, SIGMA_START, SIGMA_FINAL)

        # plot
        if (episode+1) % NUM_EPISODES_PLOT == 0:
            plt.plot(range(episode+1), rewards[:episode+1], "b")
            plt.axis([0, episode, np.min(rewards[:episode+1]), np.max(rewards[:episode+1])])
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.pause(0.1)

            if RENDER_POLICY:
                render_policy(env, theta)

    return w, theta, rewards

def render_policy(env, theta):
    """
    It shows the current learned behaviour on the GUI
    """
    env.reset()
    env.render()

    for t in range(MAX_T):
        s = get_state(env)
        a = policy(env,s,theta)
        _, reward, done, _ = env.step([a])
        env.render()

        if done:
            print("I've reached the goal!")
            break

    print("Policy executed.")

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")

    env.reset()
    env_dim = len(get_state(env))
    rewards = np.zeros(NUM_EPISODES)

    w = np.zeros(env_dim)
    theta = np.zeros(env_dim)

    if SIGMA_DECAY:
        SIGMA = SIGMA_START
    else:
        SIGMA = SIGMA_FINAL

    w, theta, rewards = training(env, w, theta, rewards)

    print("Execute final policy...")
    render_policy(env, theta)
    print("Everything is done!")

    env.close()
    plt.show()
