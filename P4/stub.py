# Imports.
import numpy as np
import numpy.random as npr

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
        self.LEARNING_RATE = 0.9
        self.DISCOUNT = 0.9
        self.A = [0, 1]
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.q = {}

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.q = {}

    def discrete_state(self, state):
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']
        return s

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.
        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.

        # At state s, take action a, end up in state s'
        s = self.discrete_state(state)
        a = self.last_action
        s_prime = self.discrete_state(self.last_state)

        if s,a not in self.q:
            self.q[s,a] = 0
        max_q = np.max(self.q[s_prime, 0], self.q[s_prime, 1])
        self.q[s,a] = (1 - self.LEARNING_RATE) * self.q[s,a] + self.LEARNING_RATE( self.last_reward + self.DISCOUNT * max_q)
        
        new_action = np.argmax(self.q[s_prime,0], self.q[s_prime,1])
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
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
        learner.reset()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Random()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


