#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:56:40 2019

@author: hufx

Example 5.1: Blackjack
"""
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm

ACTION_HIT = 0
ACTION_STICK = 1

def run_one_episode(pi):
    """
    pi: policy to estimate
    """
    trajectory = []
    
    # generate an episode following pi
    player_sum = 0
    player_usable_ace = False
    while player_sum < 12:
        card = min(numpy.random.randint(1, 14), 10)
        player_sum += 11 if card == 1 else card
        if player_sum > 21:
            assert player_sum == 22 # there may be one or two aces, and last one is ace
            player_sum -= 10
        else:
            player_usable_ace |= (card == 1)
    
    dealer_sum = 0
    dealer_card = min(numpy.random.randint(1, 14), 10)
    card = dealer_card
    while dealer_sum < 17:
        dealer_sum += 11 if card == 1 else card
        if dealer_sum > 21 and card == 1:
            dealer_sum -= 10    
        card = min(numpy.random.randint(1, 14), 10)
    
    A = pi[player_sum - 12, dealer_card - 1, int(player_usable_ace)]
    trajectory.append(((player_sum, dealer_card, player_usable_ace), A))
    while A == ACTION_HIT:
        card = min(numpy.random.randint(1, 14), 10)
        player_sum += card # ace must always be taken as numerical number 1 since initial player_sum is at least 12
        if player_sum > 21 and player_usable_ace:
            player_sum -= 10
            player_usable_ace = False
        if player_sum > 21:
            break
        A = pi[player_sum - 12, dealer_card - 1, int(player_usable_ace)]
        trajectory.append(((player_sum, dealer_card, player_usable_ace), A))
    
    if player_sum > 21:
        reward = -1
    elif dealer_sum > 21:
        reward = 1
    else:
        if player_sum > dealer_sum:
            reward = 1
        elif player_sum < dealer_sum:
            reward = -1
        else:
            reward = 0
    return reward, trajectory

def run_monte_carlo_first_visit(episodes):
    states = (10, 10, 2)
    pi = numpy.full(states, ACTION_HIT)
    pi[-2:, :, :] = ACTION_STICK
    state_values_usable_ace = numpy.zeros(states[:2])
    state_counts_usable_ace = numpy.zeros(states[:2], dtype = numpy.int)
    state_values_no_usable_ace = numpy.zeros(states[:2])
    state_counts_no_usable_ace = numpy.zeros(states[:2], dtype = numpy.int)
    for eps in tqdm(range(episodes)):
        reward, trajectory = run_one_episode(pi)
        state_visit = numpy.zeros(states, dtype = numpy.bool)
        for (player_sum, dealer_card, player_usable_ace), A in trajectory:
            if state_visit[player_sum - 12, dealer_card - 1, int(player_usable_ace)]:
                continue
            if player_usable_ace:
                state_counts_usable_ace[player_sum - 12, dealer_card - 1] += 1
                state_values_usable_ace[player_sum - 12, dealer_card - 1] += reward
            else:
                state_counts_no_usable_ace[player_sum - 12, dealer_card - 1] += 1
                state_values_no_usable_ace[player_sum - 12, dealer_card - 1] += reward
    plot(states, episodes, state_values_usable_ace, state_values_no_usable_ace, state_counts_usable_ace, state_counts_no_usable_ace, 'first_visit')

def run_monte_carlo_every_visit(episodes):
    states = (10, 10, 2)
    pi = numpy.full(states, ACTION_HIT)
    pi[-2:, :, :] = ACTION_STICK
    state_values_usable_ace = numpy.zeros(states[:2])
    state_counts_usable_ace = numpy.zeros(states[:2], dtype = numpy.int)
    state_values_no_usable_ace = numpy.zeros(states[:2])
    state_counts_no_usable_ace = numpy.zeros(states[:2], dtype = numpy.int)
    for eps in tqdm(range(episodes)):
        reward, trajectory = run_one_episode(pi)
        for (player_sum, dealer_card, player_usable_ace), A in trajectory:
            if player_usable_ace:
                state_counts_usable_ace[player_sum - 12, dealer_card - 1] += 1
                state_values_usable_ace[player_sum - 12, dealer_card - 1] += reward
            else:
                state_counts_no_usable_ace[player_sum - 12, dealer_card - 1] += 1
                state_values_no_usable_ace[player_sum - 12, dealer_card - 1] += reward
    plot(states, episodes, state_values_usable_ace, state_values_no_usable_ace, state_counts_usable_ace, state_counts_no_usable_ace, 'every_visit')

def plot(states, episodes, state_values_usable_ace, state_values_no_usable_ace, state_counts_usable_ace, state_counts_no_usable_ace, monte_carlo_method):
    xv, yv = numpy.meshgrid(numpy.arange(states[1]), numpy.arange(states[0]))
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot_wireframe(xv, yv, state_values_usable_ace / state_counts_usable_ace, rstride = 1, cstride = 1, colors = 'k')
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_zlim(-1, 1)
    ax.set_xticks([0, states[1] - 1])
    ax.set_yticks([0, states[0] - 1])
    ax.set_zticks([-1, 1])
    ax.set_xticklabels(['A', 10])
    ax.set_yticklabels([12, 21])
    ax.set_zticklabels([-1, 1])
    fig.savefig('ex13_blackjack_usable_ace_after_%d_episodes_%s_mc.png' %(episodes, monte_carlo_method))
    fig = plt.figure()
    ax = plt.gca(projection = '3d')
    ax.plot_wireframe(xv, yv, state_values_no_usable_ace / state_counts_no_usable_ace, rstride = 1, cstride = 1, colors = 'k')
    ax.set_xlabel('Dealer showing')
    ax.set_ylabel('Player sum')
    ax.set_xticks([0, states[1] - 1])
    ax.set_yticks([0, states[0] - 1])
    ax.set_zticks([-1, 1])
    ax.set_xticklabels(['A', 10])
    ax.set_yticklabels([12, 21])
    fig.savefig('ex13_blackjack_no_usable_ace_after_%d_episodes_%s_mc.png' %(episodes, monte_carlo_method))
    plt.close('all')
    
if __name__ == "__main__":
    run_monte_carlo_first_visit(10000)
    run_monte_carlo_first_visit(500000)
    run_monte_carlo_every_visit(10000)
    run_monte_carlo_every_visit(500000)