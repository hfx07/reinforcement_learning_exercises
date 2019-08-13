#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:45:45 2019

@author: hufx

Section 2.8 Gradient bandit algorithm
"""
from matplotlib import pyplot as plt
import numpy
from tqdm import tqdm

def act(levers, actions, Q, epsilon, N, t, c):
    if epsilon == 'UCB':
        if numpy.any(N == 0):
            a = numpy.random.choice(numpy.argwhere(N == 0).flatten())
        else:
            Q_ = Q + c * numpy.sqrt(numpy.log(t) / N)
            a = numpy.random.choice(numpy.argwhere(Q_ == Q_.max()).flatten())
        return a
    elif epsilon == 'gradient':
        # Q is regarded as action preference H
        pi = numpy.exp(Q) / numpy.exp(Q).sum()
        a = numpy.random.choice(actions, p = pi)
        return pi, a
    else:
        actType = numpy.random.choice(['Exploit', 'Explore'], p = [1 - epsilon, epsilon])
        if actType == 'Exploit':
            a = numpy.random.choice(numpy.argwhere(Q == Q.max()).flatten())
        else:
            a = numpy.random.randint(levers)
        return a

def run(levers, initQ, q, timeSteps, epsilon, stepSize, c, withBaseLine):
    # actions
    actions = numpy.arange(levers)
    # initial estimates of action-values
    Q = numpy.ones(levers) * initQ
    # initial action counts
    N = numpy.zeros(levers, dtype = numpy.int)
    # actions along time-steps
    A = numpy.zeros(timeSteps, dtype = numpy.int)
    # rewards along time-steps
    R = numpy.zeros(timeSteps)
    if epsilon == 'gradient':
        if withBaseLine:
            R_average = 0.
            for t in range(timeSteps):
                pi, a = act(levers, actions, Q, epsilon, N, t + 1, c)
                A[t] = a
                R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
                N[a] += 1
                # update action preference
                Q[a] += stepSize * (R[t] - R_average) * (1 - pi[a])
                Q[actions != a] -= stepSize * (R[t] - R_average) * pi[actions != a]
                R_average = (R_average * t + R[t]) / (t + 1)
        else:
            for t in range(timeSteps):
                pi, a = act(levers, actions, Q, epsilon, N, t + 1, c)
                A[t] = a
                R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
                N[a] += 1
                # update action preference
                Q[a] += stepSize * R[t] * (1 - pi[a])
                Q[actions != a] -= stepSize * R[t] * pi[actions != a]
    else:
        if stepSize is None:
            for t in range(timeSteps):
                a = act(levers, actions, Q, epsilon, N, t + 1, c)
                A[t] = a
                R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
                # incremental implementation of sample averages
                N[a] += 1
                Q[a] = Q[a] + (R[t] - Q[a]) / N[a]
        else:
            for t in range(timeSteps):
                a = act(levers, actions, Q, epsilon, N, t + 1, c)
                A[t] = a
                R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
                # incremental implementation of constant step-size
                N[a] += 1
                Q[a] = Q[a] + (R[t] - Q[a]) * stepSize
    return A, R

def test(runs, levers, timeSteps, epsilons, stepSizes, initQs, c, withBaseLines):
    average_rewards = numpy.zeros((len(withBaseLines), timeSteps))
    optimal_actions = numpy.zeros((len(withBaseLines), timeSteps), dtype = numpy.int)
    for j in tqdm(range(runs)):
        # true values of actions ~ N(4, 1)
        q = numpy.random.randn(levers) + 4
        optimalAction = numpy.argmax(q)
        for i, epsilon, stepSize, initQ, withBaseLine in zip(range(len(withBaseLines)), epsilons, stepSizes, initQs, withBaseLines):
            A, R = run(levers, initQ, q, timeSteps, epsilon, stepSize, c, withBaseLine)
            average_rewards[i, :] = average_rewards[i, :] + R
            optimal_actions[i, :] = optimal_actions[i, :] + (A == optimalAction)
    average_rewards = average_rewards / runs
    optimal_actions = optimal_actions * 1. / runs
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[0, :], color = 'blue', label = 'alpha = %s, %s baseline' %(stepSizes[0], 'with' if withBaseLines[0] else 'without'))
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[1, :], color = 'lightblue', label = 'alpha = %s, %s baseline' %(stepSizes[1], 'with' if withBaseLines[1] else 'without'))
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[2, :], color = 'goldenrod', label = 'alpha = %s, %s baseline' %(stepSizes[2], 'with' if withBaseLines[2] else 'without'))
    plt.plot(numpy.arange(1, timeSteps + 1), average_rewards[3, :], color = 'palegoldenrod', label = 'alpha = %s, %s baseline' %(stepSizes[3], 'with' if withBaseLines[3] else 'without'))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    fig.savefig('ex5_ten_armed_bandit_testbed_gradient_fig1.png')
    fig = plt.figure()
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[0, :], color = 'blue', label = 'alpha = %s, %s baseline' %(stepSizes[0], 'with' if withBaseLines[0] else 'without'))
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[1, :], color = 'lightblue', label = 'alpha = %s, %s baseline' %(stepSizes[1], 'with' if withBaseLines[1] else 'without'))
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[2, :], color = 'goldenrod', label = 'alpha = %s, %s baseline' %(stepSizes[2], 'with' if withBaseLines[2] else 'without'))
    plt.plot(numpy.arange(1, timeSteps + 1), optimal_actions[3, :], color = 'palegoldenrod', label = 'alpha = %s, %s baseline' %(stepSizes[3], 'with' if withBaseLines[3] else 'without'))
    plt.xlabel('Steps')
    plt.ylabel('Optimal action')
    plt.legend()
    fig.savefig('ex5_ten_armed_bandit_testbed_gradient_fig2.png')
    plt.close('all')

if __name__ == "__main__":
    # stepSize of None means sample-average method
    test(2000, 10, 1000, ['gradient', 'gradient', 'gradient', 'gradient'], [0.1, 0.4, 0.1, 0.4], [0, 0, 0, 0], None, [True, True, False, False])
