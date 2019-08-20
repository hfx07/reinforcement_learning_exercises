#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:57:00 2019

@author: hufx

Example 4.1: Gridworld
"""
import numpy

def policy_evaluation():
    actions = numpy.array(['up', 'down', 'right', 'left'])
    states = numpy.arange(1, 15)
    states_plus = numpy.concatenate((states, numpy.zeros(1, dtype = numpy.int))) # 0 indicates terminal state
    
    rewards = -1 * numpy.ones(states.shape + actions.shape + states_plus.shape, dtype = numpy.int)
    
    # environment dynamics
    p = numpy.zeros(states.shape + actions.shape + states_plus.shape)
    for s, sta in enumerate(states):
        if sta in [1, 2, 3]:
            p[s, 0, s] = 1
        elif sta == 4:
            p[s, 0, -1] = 1
        else:
            p[s, 0, s - 4] = 1
        if sta in [12, 13, 14]:
            p[s, 1, s] = 1
        elif sta == 11:
            p[s, 1, -1] = 1
        else:
            p[s, 1, s + 4] = 1
        if sta in [3, 7, 11]:
            p[s, 2, s] = 1
        elif sta == 14:
            p[s, 2, -1] = 1
        else:
            p[s, 2, s + 1] = 1
        if sta in [4, 8, 12]:
            p[s, 3, s] = 1
        elif sta == 1:
            p[s, 3, -1] = 1
        else:
            p[s, 3, s - 1] = 1
    # discount rate
    gamma = 1.0
    # policy (equiprobable random)
    pi = numpy.ones(states.shape + actions.shape) / actions.size
    
    # threshold
    theta = 1e-3
    # iterative policy evaluation
    V = numpy.zeros(states_plus.size)
    #V_copy = V.copy()
    delta = theta
    k = 0
    print 'v(k = %d) for the random policy:' %k
    print numpy.concatenate((V[-1:], V)).reshape((4, 4))
    while delta >= theta:
        delta = 0.0
        #for s in range(states.size):
        #    V_copy[s] = V[s]
        for s in range(states.size):
            v = V[s]
            # in-place algorithm
            V[s] = sum([pi[s, a] * sum([p[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * V[s_prime]) for s_prime in range(states_plus.size)]) for a in range(actions.size)])
            # two-array algorithm
            #V[s] = sum([pi[s, a] * sum([p[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * V_copy[s_prime]) for s_prime in range(states_plus.size)]) for a in range(actions.size)])
            delta = max(delta, abs(v - V[s]))
        k += 1
        print 'v(k = %d) for the random policy:' %k
        print numpy.concatenate((V[-1:], V)).reshape((4, 4))

if __name__ == "__main__":
    policy_evaluation()