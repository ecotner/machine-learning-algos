#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:28:47 2017

Solving blackjack via Monte Carlo with Exploring Starts (MCES), as in example 5.3 of Sutton/Barto.

The MCES algorithm is as follows:
--------------------------------------
Initialize for all s \in S, a \in A(S)
    Q(s,a) <-- arbitrary
    \pi(s) <-- arbitrary
    Returns(s,a) <-- empty list

Repeat forever:
    Randomly select starting state S0 \in S, A0 \in A(S0)
    Generate an episode according to policy \pi
    For each pair (s,a) in episode:
        G <-- return of episode following first occurence of (s,a)
        Append G to Returns(s,a)
        Q(s,a) <-- average(Returns(s,a))
    For each s in episode:
        \pi(s) <-- argmax_a(Q(s,a))
--------------------------------------

The states are indexed by 1) the dealer's card shown face-up 2) the player's current score 3) whether the player has a "usable ace"

***Turns out the first algorithm I implemented was actually a regular MC, not MCES. Even after 1e6 episodes, a large fraction of states were only visited once because they were so unlikely, leading to bad MC estimates

@author: ecotner
"""

import numpy as np
import matplotlib.pyplot as plt

class Blackjack(object):
    '''
    Blackjack game environment. Keeps track of current score, which cards of the dealer's are showing, etc. Provides methods to update the state of the game. Assumes the deck is infinite with replacement (for simplicity), so there is no advantage to counting cards.
    '''
    
    
    
    def __init__(self):
        self.usable_ace = {'player':False, 'dealer':False}
        self.hand = {'player':[], 'dealer':[]}
        self.score = {'player':0, 'dealer':0}
        self.dealer_hits_on = 16
        self.ace_prob = [1/13,12/13]
        self.face_prob = [3/12,9/12]
        self.MC_card_prob = np.array([1,1,1,1,1,1,1,1,4])/12
    
    def new_game(self):
        self.usable_ace = {'player':False, 'dealer':False}
        self.hand = {'player':[], 'dealer':[]}
        self.score = {'player':0, 'dealer':0}
    
    def deal_hand(self):
        self.new_game()
        self.hit('player')
        self.hit('player')
        self.hit('dealer')
    
    def deal_hand_MCES(self):
        ''' In order to avoid the possibility that some starting states have very low probability of being dealt in an actual game, I'm going to manipulate the stating states so that there is a uniform probability of any starting state. '''
        self.new_game()
        # Determine player hand
        self.usable_ace['player'] = np.random.choice([True,False])
        if self.usable_ace['player']:
            self.score['player'] = np.random.randint(12-10,21-10+1)
        else:
            self.score['player'] = np.random.randint(12,21+1)
        # Determine dealer hand
        self.score['dealer'] = np.random.randint(1,10+1)
        if self.score['dealer'] == 1:
            self.usable_ace['dealer'] = True
        # Make sure hand has correct length, doesn't matter what 'cards' are
        self.hand['player'] = [0,0]
        self.hand['dealer'] = [0]
    
    def hit(self, target):
        suit = np.random.choice(['\u2665','\u2666','\u2663','\u2660']) # Heart, diamond, club, spade
        # Determine whether ace drawn and if usable
        is_ace = np.random.choice([True,False], p=self.ace_prob)
        if is_ace:
            self.hand[target].append('A'+suit)
            if self.usable_ace[target]:
                if self.get_total_score(target) >= 21:
                    self.usable_ace[target] = False
            else:
                if self.score[target] <= 10:
                    self.usable_ace[target] = True
            self.score[target] += 1
        else:
            #Determine whether face or normal card
            is_face = np.random.choice([True,False], p=self.face_prob)
            if is_face:
                card = np.random.choice(['J','Q','K'])
                self.hand[target].append(card+suit)
                self.score[target] += 10
            else:
                card = np.random.randint(2,11)
                self.hand[target].append(str(card)+suit)
                self.score[target] += card
            if (self.get_total_score(target) > 21):
                self.usable_ace[target] = False
    
    def hitMC(self, target):
        ''' A less computationally-expensive hit operation that doesn't care about the hand, only numerical values '''
        # Determine whether ace drawn and if usable
        is_ace = np.random.choice([True,False], p=self.ace_prob)
        if is_ace:
            if self.usable_ace[target]:
                if self.get_total_score(target) >= 21:
                    self.usable_ace[target] = False
            else:
                if self.score[target] <= 10:
                    self.usable_ace[target] = True
            self.score[target] += 1
        # Determine what type of non-ace
        else:
            card = np.random.choice([2,3,4,5,6,7,8,9,10], p=self.MC_card_prob)
            self.score[target] += card
            if (self.get_total_score(target) > 21):
                self.usable_ace[target] = False
        # Append dummy card to hand
        self.hand[target].append(0)
    
    def get_total_score(self, target):
        ''' Keep score for usable ace separate from score for other cards '''
        if self.usable_ace[target]:
            return self.score[target]+10
        else:
            return self.score[target]
    
    def check_terminal_state(self, target):
        ''' Returns whether the player is in a terminal state, and whether that state is a bust or not '''
        s, stot = (self.score[target], self.get_total_score(target))
        assert s <= stot, "How is total score less than raw score??"
        if target == 'player':
            if s > 21:
                return True, True
            elif s < 21:
                return False, False
            elif (s == stot == 21):
                return False, False
    #        elif (s < 21) and (stot != 21):
    #            return False, False
    #        elif stot == 21:
    #            return True, False
            else:
                raise Exception("Unexpected error in self.check_terminal_state(player)")
        elif target == 'dealer':
            # Dealer has to stop if over the limit
            if stot > self.dealer_hits_on:
                # Busts if raw score over 21, otherwise safe, but still has to stop
                if s > 21:
                    return True, True
                else:
                    return True, False
            else:
                return False, False
        else:
            raise Exception("Target neither player nor dealer in check_terminal_state")
    
    def return_reward(self, is_bust, is_bust_dealer):
        # If you bust, you lose
        if is_bust:
            return -1
        elif is_bust_dealer:
            return 1
        # If one score is higher than other, outcome is unambiguous
        if self.get_total_score('player') > self.get_total_score('dealer'):
            return 1
        elif self.get_total_score('player') < self.get_total_score('dealer'):
            return -1
        # Determine what to do if score is same
        else:
            # Determine if either hand is natural (contains just ace and face card)
            is_natural = {'player':False, 'dealer':False}
            for target in is_natural:
                if (len(self.hand[target]) == 2) and (self.usable_ace[target] == True):
                    is_natural[target] = True
            # Obviously, (natural hand) > (not natural hand)
            if is_natural['player'] and (not is_natural['dealer']):
                return 1
            elif is_natural['dealer'] and (not is_natural['player']):
                return -1
            else:
                return 0

    def __str__(self):
        ''' Returns a string whose output is the visible hand and score of both dealer and player. '''
        string = 'Dealer:'
        for card in self.hand['dealer']:
            string += ' '+card
        if len(self.hand['dealer']) == 1:
            string += ' ??'
        string += '\tScore: {}'.format(self.score['dealer'])
        if self.usable_ace['dealer']:
            string += '(+10)={}'.format(self.score['dealer']+10)
        string += '\nPlayer:'
        for card in self.hand['player']:
            string += ' '+card
        string += '\tScore: {}'.format(self.score['player'])
        if self.usable_ace['player']:
            string += '(+10)={}'.format(self.score['player']+10)
        return string

    def play_blackjack(self):
        ''' Allows the user to play against the computer (dealer). '''
        win_loss_draw = [0,0,0]
        # Loop over individual games
        play_again = True
        while play_again:
            self.deal_hand()
            is_terminal_state = False
            # Loop over the player's turns
            while is_terminal_state == False:
                print(self)
                valid_string = False
                while valid_string == False:
                    is_hit = input('Hit? [Y/n]: ')
                    if is_hit.lower() in ['y','yes','','\n','hit']:
                        is_hit = True
                        valid_string = True
                    elif is_hit.lower() in ['n','no','stay']:
                        is_hit = False
                        valid_string = True
                if is_hit:
                    self.hit('player')
                    is_terminal_state, is_bust = self.check_terminal_state('player')
                else:
                    is_terminal_state = True
                    is_bust = False
            # Loop over dealer's turn if not busted
            if (not is_bust):
                is_terminal_state = False
                while is_terminal_state == False:
                    self.hit('dealer')
                    is_terminal_state, is_bust_dealer = self.check_terminal_state('dealer')
            else:
                is_bust_dealer = False
            print(self)
            # Determine outcome
            reward = self.return_reward(is_bust, is_bust_dealer)
            if reward == 1:
                print('Congratulations, you won!')
                win_loss_draw[0] += 1
            elif reward == -1:
                print('Sorry, you lost...')
                win_loss_draw[1] += 1
            else:
                print('It\'s a draw!')
                win_loss_draw[2] += 1
            print('Current score: wins: {0[0]}, losses: {0[1]}, draws: {0[2]}'.format(win_loss_draw))
            valid_string = False
            while (not valid_string):
                play_again = input('Play again? [Y/n]: ')
                if play_again.lower() in ['y','yes','','\n']:
                    play_again = True
                    valid_string = True
                elif play_again.lower() in ['n', 'no']:
                    play_again = False
                    valid_string = True

def MC_BJ(max_episodes, algo='epsilon-greedy'):
    ''' Monte Carlo algoritm for performing GPI (Generalized Policy Iteration) on the blackjack environment. '''
    troubleshooting = False
    # Enumeration of all possible states/actions
    S1 = list(range(12,21+1)) # Score of player's hand
    S2 = list(range(1,10+1)) # Cards showing in dealer's hand (10 and face cards treated the same, ace is 1)
    S3 = [0,1] # Whether or not player has a usable ace
    A = [0,1] # Whether or not to hit (0=stay, 1=hit)
    epsilon = 0.5 # Probability to take random action, linearly annealed to zero
    # Set up environment
    env = Blackjack()
    # Initialize policy \pi, action value function Q, and lists of accumlated rewards
    pi = {} # Will assume default policy is to randomly choose hit/stay
    Q = {}
    Rewards = {}
    # Repeat policy iteration loop until approx. convergence
    for episode in range(max_episodes):
        if ((episode+1) % 500 == 0):
            print('Episode {}/{}'.format(episode+1, max_episodes))
        # Pick starting state (use deal_hand_MCES for Exploring Starts)
#        env.deal_hand()
        if algo == 'epsilon-greedy':
            env.deal_hand()
        elif algo == 'exploring-starts':
            env.deal_hand_MCES()
        # Always hit if score < 12 since you can't bust
        while env.get_total_score('player') < 12:
            if troubleshooting:
                env.hit('player')
            else:
                env.hitMC('player')
        # Generate episode via MC (saving encountered states and cumulative rewards (reward is only given at the end, so keeping track of rewards is unnecessary))
        is_terminal_state = False
        encountered_states = []
        # Iterate over player's turn
        step = 0
        while (not is_terminal_state):
            s = (env.get_total_score('player'), env.score['dealer'], int(env.usable_ace['player']))
            # Choose action according to epsilon-greedy policy
            if algo == 'epsilon-greedy':
                if np.random.rand() < epsilon*(1-(episode+1)/max_episodes):
                    a = np.random.choice(A)
                else:
                    a = pi.get(s, 1)
            elif algo == 'exploring-starts':
                if step == 0:
                    a = np.random.choice(A)
                else:
                    a = pi.get(s, 1)
                step += 1                
            encountered_states.append((s,a))
            if a:
                if troubleshooting:
                    troubleshoot_before_hit(env)
                    env.hit('player')
                    troubleshoot_after_hit(env)
                else:
                    env.hitMC('player')
                is_terminal_state, is_bust = env.check_terminal_state('player')
            else:
                is_terminal_state = True
                is_bust = False
        # Iterate over dealer's turn if not busted
        if (not is_bust):
            is_terminal_state = False
            while is_terminal_state == False:
                if troubleshooting:
                    troubleshoot_before_hit(env, 'dealer')
                    env.hit('dealer')
                    troubleshoot_after_hit(env, 'dealer')
                else:
                    env.hitMC('dealer')
                is_terminal_state, is_bust_dealer = env.check_terminal_state('dealer')
        else:
            is_bust_dealer = False
        # Calculate reward for episode
        r = env.return_reward(is_bust, is_bust_dealer)
        # Calculate estimate of Q (by iterating over encountered states)
        for s, a in encountered_states:
            # Extract cumulative return after each (s,a) (all the same in this case since only terminal state gives rewards)
            # Add return to list of accumulated rewards
            Rewards[(s,a)] = Rewards.get((s,a), []) + [r]
#            Rewards.get((s,a), []).append(r)
            # Calculate Q as average of accumulated rewards
            Q[(s,a)] = np.mean(Rewards[(s,a)])
        # Calculate estimate of improved \pi(s) by taking argmax_a[Q(s,a)] (only need to update states encountered in episode though)
        for s, _ in encountered_states:
            pi[s] = np.argmax([Q.get((s,0), 100), Q.get((s,1), 100)]) # If haven't encountered a given Q(s,a) yet, give it an optimistic return so it tries it out next time
    # Plot the policy
    plot_policy(pi)
    # Count fraction of state/action pairs visited
    n_visits = 10
    total = {i:0 for i in range(n_visits+1)}
    for key in Rewards:
        n = len(Rewards[key])
        for m in range(n,n_visits+1):
            total[m] += 1
    for n in range(n_visits+1):
        print('Fraction of state/action pairs only visited {} times or less: {}'.format(n, total[n]/400))
    # Return estimates of \pi_* and Q_*
    return pi, Q

def plot_policy(pi):
    # Enumeration of all possible states/actions
    S1 = list(range(12,21+1)) # Score of player's hand
    S2 = list(range(1,10+1)) # Cards showing in dealer's hand (10 and face cards treated the same, ace is 1)
    S3 = [0,1] # Whether or not player has a usable ace
    A = [0,1] # Whether or not to hit (0=stay, 1=hit)
    # Convert dictionary to arrays
    X, Y = (S2, S1)
    Z0 = np.zeros((len(S2), len(S1)))
    Z1 = Z0.copy()
    for s2, i in zip(S2,range(len(S2))):
        for s1, j in zip(S1,range(len(S1))):
           Z0[j,i] = pi.get(((s1,s2,0)), -1)
           Z1[j,i] = pi.get(((s1,s2,1)), -1)
    contours = np.arange(-1,2) + 0.5
    # Usable ace policy
    plt.figure('pi1')
    c = plt.contourf(X, Y, Z1, contours)
    plt.title('$\pi_*$ for usable ace')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player\'s score')
    plt.colorbar(c)
    plt.show()
#    plt.savefig('pi_usable_ace.png', bbox_inches='tight')
    # No usable ace policy
    plt.figure('pi0')
    c = plt.contourf(X, Y, Z0, contours)
    plt.title('$\pi_*$ for NONusable ace')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player\'s score')
    plt.colorbar(c)
    plt.show()
#    plt.savefig('pi_NONusable_ace.png', bbox_inches='tight')

def troubleshoot_before_hit(env, target='player'):
    error_flag = False
    if env.get_total_score(target) > 21:
        print('Raw score greater than 21 on nonterminal state')
        error_flag = True
#    n_aces = len({'A\u2665','A\u2666','A\u2663','A\u2660'}.intersection(env.hand['player']))
#    if n_aces > 1:
#        print('Found lots of aces!')
#        error_flag = True
    if error_flag:
        print(env)

def troubleshoot_after_hit(env, target='player'):
    error_flag = False
    terminal_state, is_bust = env.check_terminal_state(target)
    if (env.get_total_score(target) > 21) and (not is_bust):
        print('Not busting when over 21')
        error_flag = True
    if error_flag:
        print(env)

#env = Blackjack()
#env.play_blackjack()

max_episodes = int(float(input('Max episodes: ')))
pi, Q = MC_BJ(max_episodes=max_episodes, algo='exploring-starts')

























