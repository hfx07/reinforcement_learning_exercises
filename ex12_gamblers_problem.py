#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:42:13 2019

@author: hufx

Example 4.3: Gambler's Problem
"""
import numpy
from matplotlib import pyplot as plt

def value_iteration(prob_h, gamma = 1.0, theta = 1e-20):
    states = numpy.arange(1, 100)
    states_plus = numpy.arange(1, 101)
    
    fig = plt.figure()
    # value iteration
    V = numpy.zeros(states_plus.size)
    k = 0
    delta = theta
    while delta >= theta:
        delta = 0.0
        for s in range(states.size):
            v = V[s]
            V[s] = max([prob_h * ((1 if states[s] + a == 100 else 0) + gamma * V[states[s] + a - 1]) + (1 - prob_h) * gamma * V[states[s] - a - 1] for a in range(min(states[s], 100 - states[s]) + 1)])
            delta = max(delta, abs(v - V[s]))
        k += 1
        if k == 1:
            plt.plot(states, V[:-1], 'b-', label = 'sweep %d' %k)
        elif k == 2:
            plt.plot(states, V[:-1], 'r-', label = 'sweep %d' %k)
        elif k == 3:
            plt.plot(states, V[:-1], 'g-', label = 'sweep %d' %k)
        elif k == 4:
            plt.plot(states, V[:-1], 'y-', label = 'sweep %d' %k)
    plt.plot(states, V[:-1], 'k:', label = 'Final value function')
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend()
    fig.savefig('ex12_gamblers_problem_value_estimates_ph055.png')
    
    # optimal policy
    pi_star = numpy.zeros_like(states)
    pi_star_upper = numpy.zeros_like(states)
    for s in range(states.size):
        q = numpy.round([prob_h * ((1 if states[s] + a == 100 else 0) + gamma * V[states[s] + a - 1]) + (1 - prob_h) * gamma * V[states[s] - a - 1] for a in range(min(states[s], 100 - states[s]) + 1)], 15)
        pi_star[s] = numpy.argmax(q[1:]) + 1
        pi_star_upper[s] = len(q) - 1 - numpy.argmax(q[-1::-1])
    fig  = plt.figure()
    plt.bar(states, pi_star, align = 'center', edgecolor = 'black', facecolor = 'white')
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.xticks([1, 25, 50, 75, 99])
    plt.yticks([1, 10, 20, 30, 40, 50])
    fig.savefig('ex12_gamblers_problem_final_policy_ph055.png')
    fig  = plt.figure()
    plt.bar(states, pi_star_upper, align = 'center', edgecolor = 'black', facecolor = 'white')
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.xticks([1, 25, 50, 75, 99])
    plt.yticks([1, 10, 20, 30, 40, 50])
    fig.savefig('ex12_gamblers_problem_final_policy_upperbound_ph055.png')
    plt.close('all')

if __name__ == "__main__":
    value_iteration(0.55)