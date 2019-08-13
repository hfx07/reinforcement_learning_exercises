#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:10:51 2019

@author: hufx

Example 3.3, 3.9, Exercise 3.23: Recycling robot
"""
import numpy, argparse, sys
from scipy import optimize

# r_search: expected number of cans the robot will collect while searching
# r_wait: expected number of cans the robot will collect while waiting
# alpha: probability of searching that begins with a high energy level and leaves the energy level high
# beta: probability of searching that is undertaken when the energy level is low and leaves it low
# gamma: discount rate
##r_search should be larger than r_wait
def solve(r_search, r_wait, alpha, beta, gamma):
    """
    >>> solve(1.5, 0.5, 0.5, 0.2, 0.9)
    Recycling robot optimal state-value function:
    [10.34482759  9.31034483]
    Recycling robot optimal policy:
    ['search' 'recharge']
    """
    actions = numpy.array(['search', 'wait', 'recharge'])
    states = numpy.array(['high', 'low'])
    
    rewards = numpy.zeros(states.shape + actions.shape + states.shape)
    rewards[:, 0, :] = r_search
    rewards[:, 1, :] = r_wait
    # robot must be rescued
    rewards[1, 0, 0] = -3
    
    # environment dynamics
    p = numpy.zeros(states.shape + actions.shape + states.shape)
    p[0, 0, 0] = alpha
    p[0, 0, 1] = 1 - alpha
    p[1, 0, 0] = 1 -  beta
    p[1, 0, 1] = beta
    p[0, 1, 0] = 1
    p[1, 1, 1] = 1
    p[1, 2, 0] = 1
    
    # Bellman optimality equation for state-value function
    def fun(x, actions, states):
        result = []
        for s in range(states.size):
            result.append(x[s] - max([sum([p[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * x[s_prime]) for s_prime in range(states.size)]) for a in range(actions.size)]))
        return result
    sol = optimize.root(fun, numpy.zeros(states.size), args = (actions, states), method = 'hybr')
    # optimal state-value function
    v_star = sol.x.reshape(states.shape)
    print 'Recycling robot optimal state-value function:'
    print v_star
    
    # optimal policy
    pi_star = numpy.empty(states.shape, dtype = '|S%d' %sum([len(a) + 1 for a in actions]))
    for s in range(states.size):
        q = [sum([p[s, a, s_prime] * (rewards[s, a, s_prime] + gamma * v_star[s_prime]) for s_prime in range(states.size)]) for a in range(actions.size)]
        a = numpy.argmax(q)
        pi_star[s] = actions[a]
        for k in range(actions.size):
            if k != a and numpy.abs(q[k] - q[a]) <= max(abs(q[k]), abs(q[a])) * 2 * 1.49012e-08:
                pi_star[s] += '/' + actions[k]
    print 'Recycling robot optimal policy:'
    print pi_star

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('r_search', type = float)
    parser.add_argument('r_wait', type = float)
    parser.add_argument('alpha', type = float)
    parser.add_argument('beta', type = float)
    parser.add_argument('gamma', type = float)
    argDict = dict( parser.parse_args(sys.argv[1:])._get_kwargs() )
    
    solve(argDict['r_search'], argDict['r_wait'], argDict['alpha'], argDict['beta'], argDict['gamma'])