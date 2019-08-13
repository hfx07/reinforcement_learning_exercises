#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:45:45 2019

@author: hufx

Excercise 2.11 Parameter study for nonstationary case
"""
from matplotlib import pyplot as plt
import numpy
from tqdm import tqdm
import _ex7_ten_armed_bandit_testbed_parameter_study_nonstationary as _ex7
import multiprocessing
"""
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
                # all q(a) take independent random walks
                q += 0.01 * numpy.random.randn(levers)
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
                # all q(a) take independent random walks
                q += 0.01 * numpy.random.randn(levers)
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
                # all q(a) take independent random walks
                q += 0.01 * numpy.random.randn(levers)
                R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
                # incremental implementation of sample averages
                N[a] += 1
                Q[a] = Q[a] + (R[t] - Q[a]) / N[a]
        else:
            for t in range(timeSteps):
                a = act(levers, actions, Q, epsilon, N, t + 1, c)
                A[t] = a
                # all q(a) take independent random walks
                q += 0.01 * numpy.random.randn(levers)
                R[t] = numpy.random.randn() + q[a] # ~ N(q, 1)
                # incremental implementation of constant step-size
                N[a] += 1
                Q[a] = Q[a] + (R[t] - Q[a]) * stepSize
    return A, R

def test(runs, levers, timeSteps):
    initQs_greedy = numpy.logspace(-2, 2, 5, base = 2)
    average_rewards_greedy = numpy.zeros((5, timeSteps))
    epsilons_greedy = numpy.logspace(-7, -2, 6, base = 2)
    average_rewards_epsilon_greedy = numpy.zeros((6, timeSteps))
    cs_ucb = numpy.logspace(-4, 2, 7, base = 2)
    average_rewards_ucb = numpy.zeros((7, timeSteps))
    alphas_gradient = numpy.logspace(-5, 2, 8, base = 2)
    average_rewards_gradient = numpy.zeros((8, timeSteps))
    for j in tqdm(range(runs)):
        # true values of actions: start out equal
        q = numpy.random.randn() * numpy.ones(levers)
        for i, initQ in enumerate(initQs_greedy):
            A, R = run(levers, initQ, q, timeSteps, 0, 0.1, None, None)
            average_rewards_greedy[i, :] = average_rewards_greedy[i, :] + R
        for i, epsilon in enumerate(epsilons_greedy):
            A, R = run(levers, 0, q, timeSteps, epsilon, 0.1, None, None)
            average_rewards_epsilon_greedy[i, :] = average_rewards_epsilon_greedy[i, :] + R
        for i, c in enumerate(cs_ucb):
            A, R = run(levers, 0, q, timeSteps, 'UCB', None, c, None)
            average_rewards_ucb[i, :] = average_rewards_ucb[i, :] + R
        for i, alpha in enumerate(alphas_gradient):
            A, R = run(levers, 0, q, timeSteps, 'gradient', alpha, None, True)
            average_rewards_gradient[i, :] = average_rewards_gradient[i, :] + R
    average_rewards_greedy = average_rewards_greedy / runs
    average_rewards_epsilon_greedy = average_rewards_epsilon_greedy / runs
    average_rewards_ucb = average_rewards_ucb / runs
    average_rewards_gradient = average_rewards_gradient / runs
    fig = plt.figure()
    plt.semilogx(initQs_greedy, average_rewards_greedy[:, -100000:].mean(axis = 1), basex = 2, color = 'k', label = r'greedy with optimistic initialization $\alpha = 0.1$')
    plt.semilogx(epsilons_greedy, average_rewards_epsilon_greedy[:, -100000:].mean(axis = 1), basex = 2, color = 'r', label = r'$\epsilon$-greedy $\alpha = 0.1$')
    plt.semilogx(cs_ucb, average_rewards_ucb[:, -100000:].mean(axis = 1), basex = 2, color = 'b', label = 'UCB')
    plt.semilogx(alphas_gradient, average_rewards_gradient[:, -100000:].mean(axis = 1), basex = 2, color = 'g', label = 'gradient bandit')
    plt.xlabel(r'$\epsilon$  $\alpha$  $c$  $Q_0$')
    plt.ylabel('Average reward over last 100000 steps')
    plt.legend()
    fig.savefig('ex7_ten_armed_bandit_testbed_parameter_study_nonstationary.png')
    plt.close('all')
"""

def runGreedy(qs, runs, levers, timeSteps, initQs_greedy):
    average_rewards_greedy = numpy.zeros((5, timeSteps))
    print 'runGreedy'
    for j in tqdm(range(runs)):
        for i, initQ in enumerate(initQs_greedy):
            q = qs[j].copy()
            R = _ex7.run(levers, initQ, q, timeSteps, 0.0, 0.1, False, 0.0, False, False, False)
            average_rewards_greedy[i, :] += R
    return average_rewards_greedy

def runEpsilonGreedy(qs, runs, levers, timeSteps, epsilons_greedy):
    average_rewards_epsilon_greedy = numpy.zeros((6, timeSteps))
    print 'runEpsilonGreedy'
    for j in tqdm(range(runs)):
        for i, epsilon in enumerate(epsilons_greedy):
            q = qs[j].copy()
            R = _ex7.run(levers, 0.0, q, timeSteps, epsilon, 0.1, False, 0.0, False, False, False)
            average_rewards_epsilon_greedy[i, :] += R
    return average_rewards_epsilon_greedy

def runUCB(qs, runs, levers, timeSteps, cs_ucb):
    average_rewards_ucb = numpy.zeros((7, timeSteps))
    print 'runUCB'
    for j in tqdm(range(runs)):
        for i, c in enumerate(cs_ucb):
            q = qs[j].copy()
            R = _ex7.run(levers, 0.0, q, timeSteps, 0.0, 0.0, True, c, True, False, False)
            average_rewards_ucb[i, :] += R
    return average_rewards_ucb

def runGradient(qs, runs, levers, timeSteps, alphas_gradient):
    average_rewards_gradient = numpy.zeros((8, timeSteps))
    print 'runGradient'
    for j in tqdm(range(runs)):
        for i, alpha in enumerate(alphas_gradient):
            q = qs[j].copy()
            R = _ex7.run(levers, 0.0, q, timeSteps, 0.0, alpha, False, 0.0, False, True, True)
            average_rewards_gradient[i, :] += R
    return average_rewards_gradient

def test(runs, levers, timeSteps):
    initQs_greedy = numpy.logspace(-2, 2, 5, base = 2)
    epsilons_greedy = numpy.logspace(-7, -2, 6, base = 2)
    cs_ucb = numpy.logspace(-4, 2, 7, base = 2)
    alphas_gradient = numpy.logspace(-5, 2, 8, base = 2)
    # true values of actions: start out equal
    qs = numpy.c_[numpy.random.randn(runs)] * numpy.ones((runs, levers))
    pool = multiprocessing.Pool(processes = 4)
    average_rewards_greedy = pool.apply_async(runGreedy, args = (qs, runs, levers, timeSteps,initQs_greedy))
    average_rewards_epsilon_greedy = pool.apply_async(runEpsilonGreedy, args = (qs, runs, levers, timeSteps, epsilons_greedy))
    average_rewards_ucb = pool.apply_async(runUCB, args = (qs, runs, levers, timeSteps, cs_ucb))
    average_rewards_gradient = pool.apply_async(runGradient, args = (qs, runs, levers, timeSteps, alphas_gradient))
    pool.close()
    pool.join()
    
    average_rewards_greedy = average_rewards_greedy.get() / runs
    average_rewards_epsilon_greedy = average_rewards_epsilon_greedy.get() / runs
    average_rewards_ucb = average_rewards_ucb.get() / runs
    average_rewards_gradient = average_rewards_gradient.get() / runs
    fig = plt.figure()
    plt.semilogx(initQs_greedy, average_rewards_greedy[:, -100000:].mean(axis = 1), basex = 2, color = 'k', label = r'greedy with optimistic initialization $\alpha = 0.1$')
    plt.semilogx(epsilons_greedy, average_rewards_epsilon_greedy[:, -100000:].mean(axis = 1), basex = 2, color = 'r', label = r'$\epsilon$-greedy $\alpha = 0.1$')
    plt.semilogx(cs_ucb, average_rewards_ucb[:, -100000:].mean(axis = 1), basex = 2, color = 'b', label = 'UCB')
    plt.semilogx(alphas_gradient, average_rewards_gradient[:, -100000:].mean(axis = 1), basex = 2, color = 'g', label = 'gradient bandit')
    plt.xlabel(r'$\epsilon$  $\alpha$  $c$  $Q_0$')
    plt.ylabel('Average reward over last 100000 steps')
    plt.legend()
    fig.savefig('ex7_ten_armed_bandit_testbed_parameter_study_nonstationary.png')
    plt.close('all')

if __name__ == "__main__":
    test(2000, 10, 200000)
