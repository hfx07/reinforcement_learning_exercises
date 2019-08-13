#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:04:24 2019

@author: hufx

Exercise 2.5 10-armed testbed for nonstationary problems
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

def run(q, timeSteps, epsilon, stepSize):
    # initial estimates of rewards of actions
    Q = numpy.zeros(q.size)
    # initial action counts
    N = numpy.zeros(q.size, dtype = numpy.int)
    # actions along time-steps
    A = numpy.zeros(timeSteps, dtype = numpy.int)
    # rewards along time-steps
    R = numpy.zeros(timeSteps)
    isOptimalA = numpy.zeros(timeSteps, dtype = numpy.int)
    if stepSize is None:
        for t in range(timeSteps):
            a = act(Q, epsilon)
            A[t] = a
            if t == 0 or a == numpy.argmax(q):
                isOptimalA[t] = 1
            # all q(a) take independent random walks
            q += 0.01 * numpy.random.randn(q.size)
            R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
            # incremental implementation of sample averages
            N[a] += 1
            Q[a] = Q[a] + (R[t] - Q[a]) / N[a]
    else:
        for t in range(timeSteps):
            a = act(Q, epsilon)
            A[t] = a
            if t == 0 or a == numpy.argmax(q):
                isOptimalA[t] = 1
            # all q(a) take independent random walks
            q += 0.01 * numpy.random.randn(q.size)
            R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
            # incremental implementation of constant step-size
            Q[a] = Q[a] + (R[t] - Q[a]) * stepSize
    return isOptimalA, A, R

def test(runs, levers, timeSteps, epsilon, stepSizes):
    average_rewards = numpy.zeros((len(stepSizes), timeSteps))
    optimal_actions = numpy.zeros((len(stepSizes), timeSteps), dtype = numpy.int)
    for j in tqdm(range(runs)):
        # true values of actions: start out equal
        q = numpy.random.randn() * numpy.ones(levers)
        for i, stepSize in enumerate(stepSizes):
            isOptimalA, A, R = run(q, timeSteps, epsilon, stepSize)
            average_rewards[i, :] = average_rewards[i, :] + R
            optimal_actions[i, :] = optimal_actions[i, :] + isOptimalA
    average_rewards = average_rewards / runs
    optimal_actions = optimal_actions * 1. / runs
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[0, :], 'g-', label = 'epsilon = %s (sample-average)' %epsilon)
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[1, :], 'r-', label = 'epsilon = %s (alpha = 0.1)' %epsilon)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    fig.savefig('ex2_ten_armed_bandit_testbed_nonstationary_fig1.png')
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[0, :], 'g-', label = 'epsilon = %s (sample-average)' %epsilon)
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[1, :], 'r-', label = 'epsilon = %s (alpha = 0.1)' %epsilon)
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()
    fig.savefig('ex2_ten_armed_bandit_testbed_nonstationary_fig2.png')
    plt.close('all')

if __name__ == "__main__":
    # stepSize of None means sample-average method
    test(2000, 10, 10000, 0.1, [None, 0.1])
