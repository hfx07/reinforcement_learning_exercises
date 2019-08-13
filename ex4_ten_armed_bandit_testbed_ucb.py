#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:45:45 2019

@author: hufx

Section 2.7 Upper-Confidence-Bound action selection
"""
from matplotlib import pyplot as plt
import numpy
from tqdm import tqdm

def act(Q, epsilon, N, t, c):
    if epsilon == 'UCB':
        if numpy.any(N == 0):
            a = numpy.random.choice(numpy.argwhere(N == 0).flatten())
        else:
            Q_ = Q + c * numpy.sqrt(numpy.log(t) / N)
            a = numpy.random.choice(numpy.argwhere(Q_ == Q_.max()).flatten())
    else:
        actType = numpy.random.choice(['Exploit', 'Explore'], p = [1 - epsilon, epsilon])
        if actType == 'Exploit':
            a = numpy.random.choice(numpy.argwhere(Q == Q.max()).flatten())
        else:
            a = numpy.random.randint(Q.size)
    return a

def run(initQ, q, timeSteps, epsilon, stepSize, c):
    # initial estimates of action-values
    Q = numpy.ones(q.size) * initQ
    # initial action counts
    N = numpy.zeros(q.size, dtype = numpy.int)
    # actions along time-steps
    A = numpy.zeros(timeSteps, dtype = numpy.int)
    # rewards along time-steps
    R = numpy.zeros(timeSteps)
    if stepSize is None:
        for t in range(timeSteps):
            a = act(Q, epsilon, N, t + 1, c)
            A[t] = a
            R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
            # incremental implementation of sample averages
            N[a] += 1
            Q[a] = Q[a] + (R[t] - Q[a]) / N[a]
    else:
        for t in range(timeSteps):
            a = act(Q, epsilon, N, t + 1, c)
            A[t] = a
            R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
            # incremental implementation of constant step-size
            N[a] += 1
            Q[a] = Q[a] + (R[t] - Q[a]) * stepSize
    return A, R

def test(runs, levers, timeSteps, epsilons, stepSizes, initialValueEstimates, c):
    average_rewards = numpy.zeros((len(epsilons), timeSteps))
    optimal_actions = numpy.zeros((len(epsilons), timeSteps), dtype = numpy.int)
    for j in tqdm(range(runs)):
        # true values of actions ~ N(0, 1)
        q = numpy.random.randn(levers)
        optimalAction = numpy.argmax(q)
        for i, epsilon, stepSize, initQ in zip(range(len(initialValueEstimates)), epsilons, stepSizes, initialValueEstimates):
            A, R = run(initQ, q, timeSteps, epsilon, stepSize, c)
            average_rewards[i, :] = average_rewards[i, :] + R
            optimal_actions[i, :] = optimal_actions[i, :] + (A == optimalAction)
    average_rewards = average_rewards / runs
    optimal_actions = optimal_actions * 1. / runs
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[0, :], color = 'blue', label = 'UCB c = %s' %c)
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[1, :], color = 'grey', label = 'epsilon-greedy epsilon = %s' %epsilons[1])
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    fig.savefig('ex4_ten_armed_bandit_testbed_ucb_fig1.png')
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[0, :], color = 'blue', label = 'UCB c = %s' %c)
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[1, :], color = 'grey', label = 'epsilon-greedy epsilon = %s' %epsilons[1])
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()
    fig.savefig('ex4_ten_armed_bandit_testbed_ucb_fig2.png')
    plt.close('all')

if __name__ == "__main__":
    # stepSize of None means sample-average method
    test(2000, 10, 1000, ['UCB', 0.1], [None, None], [0, 0], 2)
