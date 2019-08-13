#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:14:25 2019

@author: hufx

Section 2.3 10-armed testbed
"""
from matplotlib import pyplot as plt
import numpy
from tqdm import tqdm

def act(Q, epsilon):
    actType = numpy.random.choice(['Exploit', 'Explore'], p = [1 - epsilon, epsilon])
    if actType == 'Exploit':
        a = numpy.random.choice(numpy.argwhere(Q == Q.max()).flatten())
    else:
        a = numpy.random.randint(Q.size)
    return a

def run(q, timeSteps, epsilon):
    # initial estimates of rewards of actions
    Q = numpy.zeros(q.size)
    # initial action counts
    N = numpy.zeros(q.size, dtype = numpy.int)
    # actions along time-steps
    A = numpy.zeros(timeSteps, dtype = numpy.int)
    # rewards along time-steps
    R = numpy.zeros(timeSteps)
    for t in range(timeSteps):
        a = act(Q, epsilon)
        A[t] = a
        R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
        # incremental implementation of sample averages
        N[a] += 1
        Q[a] = Q[a] + (R[t] - Q[a]) / N[a]
    return A, R

def test(runs, levers, timeSteps, epsilons):
    average_rewards = numpy.zeros((len(epsilons), timeSteps))
    optimal_actions = numpy.zeros((len(epsilons), timeSteps), dtype = numpy.int)
    for j in tqdm(range(runs)):
        # true values of actions ~ N(0, 1)
        q = numpy.random.randn(levers)
        optimalAction = numpy.argmax(q)
        for i, epsilon in enumerate(epsilons):
            A, R = run(q, timeSteps, epsilon)
            average_rewards[i, :] = average_rewards[i, :] + R
            optimal_actions[i, :] = optimal_actions[i, :] + (A == optimalAction)
    average_rewards = average_rewards / runs
    optimal_actions = optimal_actions * 1. / runs
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[0, :], 'g-', label = 'epsilon = %s%s' %(epsilons[0], ' (greedy)' if epsilons[0] == 0 else ''))
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[1, :], 'r-', label = 'epsilon = %s%s' %(epsilons[1], ' (greedy)' if epsilons[1] == 0 else ''))
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[2, :], 'b-', label = 'epsilon = %s%s' %(epsilons[2], ' (greedy)' if epsilons[2] == 0 else ''))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    fig.savefig('ex1_ten_armed_bandit_testbed_fig1.png')
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[0, :], 'g-', label = 'epsilon = %s%s' %(epsilons[0], ' (greedy)' if epsilons[0] == 0 else ''))
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[1, :], 'r-', label = 'epsilon = %s%s' %(epsilons[1], ' (greedy)' if epsilons[1] == 0 else ''))
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[2, :], 'b-', label = 'epsilon = %s%s' %(epsilons[2], ' (greedy)' if epsilons[2] == 0 else ''))
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()
    fig.savefig('ex1_ten_armed_bandit_testbed_fig2.png')
    plt.close('all')

if __name__ == "__main__":
    fig = plt.figure()
    plt.violinplot(numpy.random.randn(200, 10) + numpy.random.randn(10), showmeans = True)
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    fig.savefig('ex1_ten_armed_bandit_testbed_fig0.png')
    test(2000, 10, 1000, [0, 0.01, 0.1])
