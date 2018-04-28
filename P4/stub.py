# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from SwingyMonkey import SwingyMonkey


class Random(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        new_action = npr.rand() < 0.1
        new_state  = state
        self.last_action = new_action
        self.last_state  = new_state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward

class QLearner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        # Constants
        self.LEARNING_RATE = 0.9
        self.DISCOUNT = 0.9
        self.A = [0, 1]
        self.SCREEN_WIDTH  = 600
        self.SCREEN_HEIGHT = 400
        # Q-learning
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.q = {}

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def discrete_state(self, state):
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']

        dist_to_tree_top = np.sqrt(tree_dist**2 + (tree_top - monkey_top)**2)
        dist_to_tree_bot = np.sqrt(tree_dist**2 + (tree_bot - monkey_bot)**2)
        pixel_info = np.array([dist_to_tree_bot, dist_to_tree_top])

        # Bin all continuous pixel ranges into groups of 100 (e.g. 400-499 -> 4, 500-599 -> 5, etc.)
        pixel_trans = np.floor_divide(pixel_info, 100).astype(int).tolist()
        # Bin velocity into groups of 10
        vel_trans = [int(monkey_vel/10)]
        trans = pixel_trans + vel_trans
        return tuple(trans)

    def q_lookup(self, s, a):
        if (s,a) not in self.q:
            return 0
        return self.q[(s,a)]

    def q_update(self, s, a, r, s_prime):
        # If we haven't seen this (s,a) pair before, add it to Q
        if (s,a) not in self.q:
            self.q[(s,a)] = 0
            print("New state ("+str(s)+", "+str(a)+") = "+str(self.q[(s,a)]))
        else:
            print("Already seen ("+str(s)+", "+str(a)+") = "+str(self.q[(s,a)]))
        # Q-learning algorithm
        max_q = np.max([self.q_lookup(s_prime, 0), self.q_lookup(s_prime, 1)])
        self.q[(s,a)] = (1 - self.LEARNING_RATE) * self.q_lookup(s,a) + self.LEARNING_RATE * ( r + self.DISCOUNT * max_q)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.
        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.

        # At state s (self.last_state), we took action a (self.last_action), got reward r (self.last_reward), and now are in state s' and need to decide a'
        s_prime = self.discrete_state(state)
        if self.last_action is not None:
            ## This new action a' will not be the very first action we take
            ## Thus, we can safely check previous state/action (s/a)
            a = self.last_action
            s = self.last_state
            r = self.last_reward
            self.q_update(s, a, r, s_prime)
            a_prime = np.argmax([self.q_lookup(s_prime, 0), self.q_lookup(s_prime, 1)])
            print("(s,a,r,s',a'): "+str(s)+","+str(a)+","+str(r)+","+str(s_prime)+","+str(a_prime))
            print(a_prime)
        
        # Get new policy from Q
        new_action = np.argmax([self.q_lookup(s_prime, 0), self.q_lookup(s_prime, 1)])
        new_state  = s_prime

        # Save current s_prime/a_prime as last s/a
        self.last_action = new_action
        self.last_state  = new_state
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):

    t_len = 10
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)
        # Loop until you hit something.
        while swing.game_loop():
            pass
        # Save score history.
        hist.append(swing.score)
        # Reset the state of the learner.
        print("---- Q ----")
        print(learner.q)
        print("\n\n")
        print("---- NEW GAME ----")
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = QLearner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


