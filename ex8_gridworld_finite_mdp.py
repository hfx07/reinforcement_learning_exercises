#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:57:00 2019

@author: hufx

Example 3.5, 3.8: Gridworld
"""
import numpy, itertools
from scipy import optimize

def calculate():
    actions = numpy.array(['north', 'south', 'east', 'west'])
    states = numpy.zeros((5, 5), dtype = numpy.int)
    
    rewards = numpy.zeros(states.shape + actions.shape)
    rewards[0, :, 0] = -1
    rewards[-1, :, 1] = -1
    rewards[:, -1, 2] = -1
    rewards[:, 0, 3] = -1
    rewards[0, 1, :] = 10
    rewards[0, 3, :] = 5
    
    # environment dynamics
    p = numpy.zeros(states.shape + actions.shape + states.shape)
    for s in itertools.product(*[range(dim) for dim in states.shape]):
        if s[0] > 0:
            p[s + (0,) + (s[0] - 1, s[1])] = 1
        else:
            p[s + (0,) + s] = 1
        if s[0] < states.shape[0] - 1:
            p[s + (1,) + (s[0] + 1, s[1])] = 1
        else:
            p[s + (1,) + (s[0], s[1])] = 1
        if s[1] < states.shape[1] - 1:
            p[s + (2,) + (s[0], s[1] + 1)] = 1
        else:
            p[s + (2,) + (s[0], s[1])] = 1
        if s[1] > 0:
            p[s + (3,) + (s[0], s[1] - 1)] = 1
        else:
            p[s + (3,) + (s[0], s[1])] = 1
    p[0, 1, :, :, :] = 0
    p[0, 1, :, 4, 1] = 1
    p[0, 3, :, :, :] = 0
    p[0, 3, :, 2, 3] = 1
    # discount rate
    gamma = 0.9
    
    # policy (equiprobable random)
    pi = numpy.ones(states.shape + actions.shape) / actions.size
    
    # solve linear equations of state-value function given policy
    matrix = numpy.zeros((states.size, states.size))
    vector = numpy.zeros(states.size)
    for i, s in enumerate(itertools.product(*[range(dim) for dim in states.shape])):
        matrix[i, i] += 1
        for a in itertools.product(*[range(dim) for dim in actions.shape]):
            for j, s_prime in enumerate(itertools.product(*[range(dim) for dim in states.shape])):
                matrix[i, j] -= pi[s + a] * p[s + a + s_prime] * gamma
                vector[i] += pi[s + a] * p[s + a + s_prime] * rewards[s + a]
    # state values
    v = numpy.linalg.solve(matrix, vector).reshape(states.shape)
    print 'Gridworld state-value function given equiprobable random policy:'
    print v

def solve():
    actions = numpy.array(['north', 'south', 'east', 'west'])
    states = numpy.zeros((5, 5), dtype = numpy.int)
    
    rewards = numpy.zeros(states.shape + actions.shape)
    rewards[0, :, 0] = -1
    rewards[-1, :, 1] = -1
    rewards[:, -1, 2] = -1
    rewards[:, 0, 3] = -1
    rewards[0, 1, :] = 10
    rewards[0, 3, :] = 5
    
    # environment dynamics
    p = numpy.zeros(states.shape + actions.shape + states.shape)
    for s in itertools.product(*[range(dim) for dim in states.shape]):
        if s[0] > 0:
            p[s + (0,) + (s[0] - 1, s[1])] = 1
        else:
            p[s + (0,) + s] = 1
        if s[0] < states.shape[0] - 1:
            p[s + (1,) + (s[0] + 1, s[1])] = 1
        else:
            p[s + (1,) + (s[0], s[1])] = 1
        if s[1] < states.shape[1] - 1:
            p[s + (2,) + (s[0], s[1] + 1)] = 1
        else:
            p[s + (2,) + (s[0], s[1])] = 1
        if s[1] > 0:
            p[s + (3,) + (s[0], s[1] - 1)] = 1
        else:
            p[s + (3,) + (s[0], s[1])] = 1
    p[0, 1, :, :, :] = 0
    p[0, 1, :, 4, 1] = 1
    p[0, 3, :, :, :] = 0
    p[0, 3, :, 2, 3] = 1
    # discount rate
    gamma = 0.9
    
    # Bellman optimality equation for state-value function
    def fun(x, actions, states):
        result = []
        for i, s in enumerate(itertools.product(*[range(dim) for dim in states.shape])):
            result.append(x[i] - max([sum([p[s + a + s_prime] * (rewards[s + a] + gamma * x[j]) for j, s_prime in enumerate(itertools.product(*[range(dim) for dim in states.shape]))]) for a in itertools.product(*[range(dim) for dim in actions.shape])]))
        return result
    sol = optimize.root(fun, numpy.zeros(states.size), args = (actions, states), method = 'hybr')
    # optimal state-value function
    v_star = sol.x.reshape(states.shape)
    print 'Gridworld optimal state-value function:'
    print v_star
    
    # optimal policy
    pi_star = numpy.empty(states.shape, dtype = '|S%d' %sum([len(a) + 1 for a in actions]))
    for s in itertools.product(*[range(dim) for dim in states.shape]):
        q = [sum([p[s + a + s_prime] * (rewards[s + a] + gamma * v_star[s_prime]) for s_prime in itertools.product(*[range(dim) for dim in states.shape])]) for a in itertools.product(*[range(dim) for dim in actions.shape])]
        a = numpy.argmax(q)
        pi_star[s] = actions[a]
        for k in range(actions.size):
            if k != a and numpy.abs(q[k] - q[a]) <= max(abs(q[k]), abs(q[a])) * 2 * 1.49012e-08:
                pi_star[s] += '/' + actions[k]
    print 'Gridworld optimal policy:'
    print pi_star

if __name__ == "__main__":
    calculate()
    solve()