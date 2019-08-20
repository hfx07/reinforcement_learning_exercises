#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:00:14 2019

@author: hufx

Example 4.2 and Exercise 4.7: Jack's Car Rental
"""
import numpy, itertools, math, os
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def policy_evaluation(pi, V, actions, states, rewards, p, discount_rate, theta):
    delta = theta
    while delta >= theta:
        delta = 0.0
        for s in itertools.product(*[range(state) for state in states]):
            v = V[s]
            a = actions.tolist().index(pi[s])
            # in-place algorithm
            V[s] = sum([p[s + (a,) + s_prime] * V[s_prime] for s_prime in itertools.product(*[range(state) for state in states])]) * discount_rate + rewards[s + (a,)]
            delta = max(delta, abs(v - V[s]))
    return V

def policy_iteration(max_available_cars, max_move_cars, rent_credit, move_cost, expected_request_cars_first, expected_request_cars_second, expected_return_cars_first, expected_return_cars_second, discount_rate, theta):
    actions = numpy.arange(-max_move_cars, max_move_cars + 1) # cars moved from first location to second location
    states = (max_available_cars + 1, max_available_cars + 1) # possible number of cars at each location at the end of each day
    
    prob_request_first = [expected_request_cars_first ** n * numpy.exp(-expected_request_cars_first) / math.factorial(n) for n in range(max_available_cars)]
    prob_request_first.append(1.0 - sum(prob_request_first))
    prob_request_second = [expected_request_cars_second ** n * numpy.exp(-expected_request_cars_second) / math.factorial(n) for n in range(max_available_cars)]
    prob_request_second.append(1.0 - sum(prob_request_second))
    prob_return_first = [expected_return_cars_first ** n * numpy.exp(-expected_return_cars_first) / math.factorial(n) for n in range(max_available_cars)]
    prob_return_first.append(1.0 - sum(prob_return_first))
    prob_return_second = [expected_return_cars_second ** n * numpy.exp(-expected_return_cars_second) / math.factorial(n) for n in range(max_available_cars)]
    prob_return_second.append(1.0 - sum(prob_return_second))
    
    # rewards r(s, a)
    if os.path.exists('ex11_jacks_car_rental_rewards.npy'):
        rewards = numpy.load('ex11_jacks_car_rental_rewards.npy')
    else:
        rewards = numpy.zeros(states + actions.shape)
        t = tqdm(total = states[0] * states[1] * actions.size * (max_available_cars + 1) * (max_available_cars + 1))
        for s in itertools.product(*[range(state) for state in states]):
            for a in range(actions.size):
                if actions[a] > s[0] or actions[a] < -s[1]:
                    t.update((max_available_cars + 1) * (max_available_cars + 1))
                    continue
                rewards[s + (a,)] -= move_cost * abs(actions[a])
                for request_cars_first, request_cars_second in itertools.product(range(max_available_cars + 1), range(max_available_cars + 1)):
                    rewards[s + (a,)] += rent_credit * (min(s[0] - actions[a], request_cars_first) + min(s[1] + actions[a], request_cars_second)) * prob_request_first[request_cars_first] * prob_request_second[request_cars_second]
                    t.update()
        numpy.save('ex11_jacks_car_rental_rewards.npy', rewards)
    
    # environment dynamics p(s'|s, a)
    if os.path.exists('ex11_jacks_car_rental_dynamics.npy'):
        p = numpy.load('ex11_jacks_car_rental_dynamics.npy')
    else:
        p = numpy.zeros(states + actions.shape + states)
        t = tqdm(total = states[0] * states[1] * actions.size * (max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1))
        for s in itertools.product(*[range(state) for state in states]):
            for a in range(actions.size):
                if actions[a] > s[0] or actions[a] < -s[1]:
                    t.update((max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1))
                    continue
                for request_cars_first, request_cars_second in itertools.product(range(max_available_cars + 1), range(max_available_cars + 1)):
                    for return_cars_first, return_cars_second in itertools.product(range(max_available_cars + 1), range(max_available_cars + 1)):
                        p[s + (a,) + (min(max_available_cars, max(0, s[0] - actions[a] - request_cars_first) + return_cars_first), min(max_available_cars, max(0, s[1] + actions[a] - request_cars_second) + return_cars_second))] += prob_request_first[request_cars_first] * prob_request_second[request_cars_second] * prob_return_first[return_cars_first] * prob_return_second[return_cars_second]
                        t.update()
        numpy.save('ex11_jacks_car_rental_dynamics.npy', p)
    
    ## policy iteration
    V = numpy.zeros(states)
    pi = numpy.zeros(states, dtype = numpy.int)
    
    # policy improvement
    k = 0
    plot_policy(actions, states, k, pi)
    policy_stable = False
    while not policy_stable:
        k += 1
        # policy evaluation
        V = policy_evaluation(pi, V, actions, states, rewards, p, discount_rate, theta)
        
        policy_stable = True
        for s in itertools.product(*[range(state) for state in states]):
            old_action = pi[s]
            q = [sum([p[s + (a,) + s_prime] * V[s_prime] for s_prime in itertools.product(*[range(state) for state in states])]) * discount_rate + rewards[s + (a,)] for a in range(actions.size)]
            pi[s] = actions[numpy.argmax(q)]
            if policy_stable is True and old_action != pi[s]:
                policy_stable = False
        
        plot_policy(actions, states, k, pi)
    
    numpy.save('ex11_jacks_car_rental_optimal_state_value_function.npy', V)
    
    xv, yv = numpy.meshgrid(numpy.arange(states[1]), numpy.arange(states[0]))
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    plt.title(r'$v_{\pi_4}$')
    ax.plot_surface(xv, yv, V, rstride = 1, cstride = 1, cmap = plt.cm.Greys)
    ax.set_xlabel('#Cars at second location')
    ax.set_ylabel('#Cars at first location')
    ax.set_xticks([0, states[1] - 1])
    ax.set_yticks([0, states[0] - 1])
    fig.savefig('ex11_jacks_car_rental_state_value_function.png')

def policy_iteration_exercise(max_available_cars, max_move_cars, rent_credit, move_cost, expected_request_cars_first, expected_request_cars_second, expected_return_cars_first, expected_return_cars_second, max_free_parking_cars, parking_cost, discount_rate, theta):
    actions = numpy.arange(-max_move_cars, max_move_cars + 1) # cars moved from first location to second location
    states = (max_available_cars + 1, max_available_cars + 1) # possible number of cars at each location at the end of each day
    
    prob_request_first = [expected_request_cars_first ** n * numpy.exp(-expected_request_cars_first) / math.factorial(n) for n in range(max_available_cars)]
    prob_request_first.append(1.0 - sum(prob_request_first))
    prob_request_second = [expected_request_cars_second ** n * numpy.exp(-expected_request_cars_second) / math.factorial(n) for n in range(max_available_cars)]
    prob_request_second.append(1.0 - sum(prob_request_second))
    prob_return_first = [expected_return_cars_first ** n * numpy.exp(-expected_return_cars_first) / math.factorial(n) for n in range(max_available_cars)]
    prob_return_first.append(1.0 - sum(prob_return_first))
    prob_return_second = [expected_return_cars_second ** n * numpy.exp(-expected_return_cars_second) / math.factorial(n) for n in range(max_available_cars)]
    prob_return_second.append(1.0 - sum(prob_return_second))
    
    # rewards r(s, a)
    if os.path.exists('ex11_jacks_car_rental_rewards_ex.npy'):
        rewards = numpy.load('ex11_jacks_car_rental_rewards_ex.npy')
    else:
        rewards = numpy.zeros(states + actions.shape)
        t = tqdm(total = states[0] * states[1] * actions.size * (max_available_cars + 1) * (max_available_cars + 1))
        for s in itertools.product(*[range(state) for state in states]):
            for a in range(actions.size):
                if actions[a] > s[0] or actions[a] < -s[1]:
                    t.update((max_available_cars + 1) * (max_available_cars + 1))
                    continue
                # consider one car shuttled to the second location for free, if any
                if actions[a] > 0:
                    rewards[s + (a,)] -= move_cost * (actions[a] - 1)
                else:
                    rewards[s + (a,)] -= move_cost * (-actions[a])
                # consider additional parking cost
                if s[0] - actions[a] > max_free_parking_cars:
                    rewards[s + (a,)] -= parking_cost
                if s[1] + actions[a] > max_free_parking_cars:
                    rewards[s + (a,)] -= parking_cost
                for request_cars_first, request_cars_second in itertools.product(range(max_available_cars + 1), range(max_available_cars + 1)):
                    rewards[s + (a,)] += rent_credit * (min(s[0] - actions[a], request_cars_first) + min(s[1] + actions[a], request_cars_second)) * prob_request_first[request_cars_first] * prob_request_second[request_cars_second]
                    t.update()
        numpy.save('ex11_jacks_car_rental_rewards_ex.npy', rewards)
    
    # environment dynamics p(s'|s, a)
    if os.path.exists('ex11_jacks_car_rental_dynamics.npy'):
        p = numpy.load('ex11_jacks_car_rental_dynamics.npy')
    else:
        p = numpy.zeros(states + actions.shape + states)
        t = tqdm(total = states[0] * states[1] * actions.size * (max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1))
        for s in itertools.product(*[range(state) for state in states]):
            for a in range(actions.size):
                if actions[a] > s[0] or actions[a] < -s[1]:
                    t.update((max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1) * (max_available_cars + 1))
                    continue
                for request_cars_first, request_cars_second in itertools.product(range(max_available_cars + 1), range(max_available_cars + 1)):
                    for return_cars_first, return_cars_second in itertools.product(range(max_available_cars + 1), range(max_available_cars + 1)):
                        p[s + (a,) + (min(max_available_cars, max(0, s[0] - actions[a] - request_cars_first) + return_cars_first), min(max_available_cars, max(0, s[1] + actions[a] - request_cars_second) + return_cars_second))] += prob_request_first[request_cars_first] * prob_request_second[request_cars_second] * prob_return_first[return_cars_first] * prob_return_second[return_cars_second]
                        t.update()
        numpy.save('ex11_jacks_car_rental_dynamics.npy', p)
    
    ## policy iteration
    V = numpy.zeros(states)
    pi = numpy.zeros(states, dtype = numpy.int)
    
    # policy improvement
    k = 0
    plot_policy(actions, states, k, pi, True)
    policy_stable = False
    while not policy_stable:
        k += 1
        # policy evaluation
        V = policy_evaluation(pi, V, actions, states, rewards, p, discount_rate, theta)
        
        policy_stable = True
        for s in itertools.product(*[range(state) for state in states]):
            old_action = pi[s]
            q = [sum([p[s + (a,) + s_prime] * V[s_prime] for s_prime in itertools.product(*[range(state) for state in states])]) * discount_rate + rewards[s + (a,)] for a in range(actions.size)]
            pi[s] = actions[numpy.argmax(q)]
            if policy_stable is True and old_action != pi[s]:
                policy_stable = False
        
        plot_policy(actions, states, k, pi, True)
    
    numpy.save('ex11_jacks_car_rental_ex_optimal_state_value_function.npy', V)
    
    xv, yv = numpy.meshgrid(numpy.arange(states[1]), numpy.arange(states[0]))
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    plt.title(r'$v_{\pi_4}$')
    ax.plot_surface(xv, yv, V, rstride = 1, cstride = 1, cmap = plt.cm.Greys)
    ax.set_xlabel('#Cars at second location')
    ax.set_ylabel('#Cars at first location')
    ax.set_xticks([0, states[1] - 1])
    ax.set_yticks([0, states[0] - 1])
    fig.savefig('ex11_jacks_car_rental_ex_optimal_state_value_function.png')
    plt.close('all')

def plot_policy(actions, states, k, pi, isExercise = False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # length of x and y arrays should be one greater than number of states, otherwise (if equal) the last row and column of pi will be ignored
    x = numpy.arange(states[1] + 1)
    y = numpy.arange(states[0] + 1)
    xv, yv = numpy.meshgrid(x, y)
    hm = ax.pcolor(xv, yv, pi)
    plt.colorbar(hm)
    plt.title(r'$\pi_{k}$'.format(k = k))
    plt.xlabel('#Cars at second location')
    plt.ylabel('#Cars at first location')
    ax.set_xticks(x, minor = False)
    ax.set_yticks(y, minor = False)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    fig.savefig('ex11_jacks_car_rental_policy_%d.png' %k if not isExercise else 'ex11_jacks_car_rental_ex_policy_%d.png' %k)
    plt.close('all')

if __name__ == "__main__":
    #policy_iteration(20, 5, 10, 2, 3, 4, 3, 2, 0.9, 1e-6)
    policy_iteration_exercise(20, 5, 10, 2, 3, 4, 3, 2, 10, 4, 0.9, 1e-6)