import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve, plot_other_curve
from my_environment import DoubleBattery
from pendulum import PendulumEnv
from stable_baselines3.common.env_checker import check_env
if __name__ == '__main__':

    env = DoubleBattery()
    check_env(env, warn=True)
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 500

    figure_file = 'plots/battery.png'
    imbalance_file  = 'plots/battery_imbalance.png'
    other_file = 'plots/battery_other.png'

    best_score = env.reward_range[0]
    score_history = []
    imbalance_history = []
    load_history = []
    soe_history = []
    action_history = []
    load_checkpoint = True

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward_norm, done, info = env.step(action)
            agent.remember(observation, action, reward_norm, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        total_imbalance = 0
        while not done:
            #env.render()
            action = agent.choose_action(observation, evaluate)
            observation_, reward_norm, done, info = env.step(action)
            reward, imbalance, load, soe, last_action = env.retrieve_data()
            score += reward
            agent.remember(observation, action, reward_norm, observation_, done)
            total_imbalance += imbalance
            
            load_history.append(load)
            soe_history.append(soe)
            action_history.append(last_action)
            
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        
        imbalance_history.append(total_imbalance)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file,1)
        
    plot_other_curve(imbalance_history, imbalance_file,2, "Imbalance")
    #plot_other_curve(load_history, other_file, 3, "Load, Actions")
    #plot_other_curve(action_history, other_file, 3, "Load, Actions")
        
        
