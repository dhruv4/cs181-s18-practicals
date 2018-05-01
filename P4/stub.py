# Imports.
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Random(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here
        # based on the current state and the last state.
        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        new_action = npr.rand() < 0.1
        new_state = state
        self.last_action = new_action
        self.last_state = new_state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


class QLearner(object):

    def __init__(self):
        # Constants - Same each game
        self.LEARNING_RATE = 0.8
        self.DISCOUNT = 1
        self.EPSILON = 0.1
        self.A = [0, 1]
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 400
        # Constant - Randomly initialized at start of each game
        self.GRAVITY = None
        self.PROP_REWARD_STATES_LENGTH = 5
        # Q-learning
        self.last_state_full = None
        self.last_state_discrete = None
        self.last_action = None
        self.last_reward = None
        self.q = {}
        # FIFO stack of (s, a) pairs to propagate reward back to
        self.prop_reward_states = []

    def reset(self):
        self.last_state_full = None
        self.last_state_discrete = None
        self.last_action = None
        self.last_reward = None
        self.GRAVITY = None

    def greedy_epsilon(self, potential_actions):
        num = np.random.uniform(0, 1)
        print ("num", num)
        if num < self.EPSILON:
            # Pick random action with probability self.EPSILON
            print ("pot act", potential_actions)
            return np.random.choice(range(len(potential_actions)))
        return np.argmax(potential_actions)

    def discrete_state(self, state):
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']

        #
        # Feature extraction
        #
        above_tree_dist = monkey_top - tree_top

        #
        # Transformations
        #

        # Bin all continuous pixel ranges
        dist_to_tree_trans = np.floor_divide(tree_dist, 50)
        if dist_to_tree_trans < 0:
            dist_to_tree_trans = 0
        dist_above_tree_trans = np.floor_divide(above_tree_dist, 50)
        # Take gravity effects on monkey velocity into account
        vel_trans = monkey_vel - (
            self.GRAVITY if self.GRAVITY is not None else 0)
        # Bin velocity
        motion_trans = int(vel_trans / 2)
        new_state = (dist_to_tree_trans, dist_above_tree_trans, motion_trans)
        return new_state

    def calc_gravity(self, current_state, last_state):
        last_monkey_vel = last_state['monkey']['vel']
        current_monkey_vel = current_state['monkey']['vel']
        self.GRAVITY = last_monkey_vel - current_monkey_vel

    def q_lookup(self, s, a):
        if (s, a) not in self.q:
            return 0
        else:
            return self.q[(s, a)]

    def q_update(self, s, a, r, s_prime):
        print("&&&&&&&&&&.   " + str(r))

        q_idx = (s, a)
        self.prop_reward_states.insert(0, q_idx)

        if q_idx not in self.q:
            self.q[q_idx] = 0
            print(
                "New state (" + str(s) + ", " + str(a) + ") = " +
                str(self.q[q_idx]))
        else:
            print(
                "Already seen (" + str(s) + ", " + str(a) + ") = " +
                str(self.q[q_idx]))

        # Q-learning algorithm
        if r < 0:
            print(": R < 0")
            # If non-zero reward, propagate back
            # to all states in self.prop_reward_states
            _s_prime = s_prime
            # Work backwards to update
            # If we have (s1, a1) -> (s2, a2) -> (s3, a3) -> (s4, a4)
            # Then set Q(s3, a3) based on (s4, 0/1),
            # Q(s2, a2) based on Q(s3, 0/1), etc.
            for (_s, _a) in self.prop_reward_states:
                max_q = np.max(
                    [self.q_lookup(_s_prime, 0), self.q_lookup(_s_prime, 1)])
                self.q[(_s, _a)] = (
                    1 - self.LEARNING_RATE) * self.q_lookup(
                    _s, _a) + self.LEARNING_RATE * (
                    r * 100 + self.DISCOUNT * max_q)
                _s_prime = _s
            self.prop_reward_states = []
        else:
            # No penalty, so give slight reward
            max_q = np.max(
                [self.q_lookup(s_prime, 0), self.q_lookup(s_prime, 1)])
            self.q[(s, a)] = (
                1 - self.LEARNING_RATE) * self.q_lookup(
                s, a) + self.LEARNING_RATE * (1 + self.DISCOUNT * max_q)
            if len(self.prop_reward_states) > self.PROP_REWARD_STATES_LENGTH:
                del self.prop_reward_states[-1]

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here
        # based on the current state and the last state.
        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.

        # At state s (self.last_state), we took action
        # a (self.last_action), got reward r (self.last_reward),
        # and now are in state s' and need to decide a'
        s_prime = self.discrete_state(state)

        if self.last_action is None:
            # first action, default to not doing anything to measure gravity
            a_prime = 0
        else:
            # Calculate gravity based on monkey position
            if self.GRAVITY is None:
                self.calc_gravity(state, self.last_state_full)
            # This new action a' will not be the very first action we take
            # Thus, we can safely check previous state/action (s/a)
            a = self.last_action
            s = self.last_state_discrete
            r = self.last_reward
            self.q_update(s, a, r, s_prime)
            a_prime = 0 if self.q_lookup(
                s_prime, 0) >= self.q_lookup(s_prime, 1) else 1
            print(
                "(s, a, r, s', a'): " + str(s) + ", " +
                str(a) + ", " + str(r) + ", " + str(s_prime) +
                ", " + str(a_prime))
            print(
                str(self.q_lookup(s_prime, 0)) + " v. " +
                str(self.q_lookup(s_prime, 1)))

        # Save current s_prime/a_prime as last s/a
        self.last_action = a_prime
        self.last_state_discrete = s_prime
        self.last_state_full = state
        return a_prime

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


class OldQLearner(object):

    def __init__(self):
        # Constants - Same each game
        self.LEARNING_RATE = 0.8
        self.DISCOUNT = 0.90
        self.EPSILON = 0.1
        self.A = [0, 1]
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 400
        self.NEAR_GROUND = np.floor_divide(100, 50)
        # Constant - Randomly initialized at start of each game
        self.gravity = None
        # Q-learning
        self.last_state_full = None
        self.last_state_discrete = None
        self.last_action = None
        self.last_reward = None
        self.q = {}

    def reset(self):
        self.last_state_full = None
        self.last_state_discrete = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    def greedy_epsilon(self, potential_actions):
        num = np.random.uniform(0, 1)
        print ("num", num)
        if num < self.EPSILON:
            # Pick random action with probability self.EPSILON
            # print ("pot act", potential_actions)
            return np.random.choice(range(len(potential_actions)))
        return np.argmax(potential_actions)

    def discrete_state(self, state):
        tree_dist = state['tree']['dist']
        tree_top = state['tree']['top']
        tree_bot = state['tree']['bot']
        monkey_vel = state['monkey']['vel']
        monkey_top = state['monkey']['top']
        monkey_bot = state['monkey']['bot']

        #
        # Feature extraction
        #
        tree_center = (tree_top - tree_bot) / 2 + tree_bot
        monkey_center = (monkey_top - monkey_bot) / 2 + monkey_bot

        dist_to_tree_center = np.sqrt(
            (tree_dist) ** 2 + (tree_center - monkey_center) ** 2)
        above_tree_center = (monkey_center > tree_center)
        # above_tree_center = (monkey_top < tree_top) and (monkey_bot > tree_bot)
        # above_tree_center = True
        dist_to_ground = monkey_center
        # dist_to_tree_top = np.sqrt(tree_dist**2 + (tree_top - monkey_top)**2)
        # dist_to_tree_bot = np.sqrt(tree_dist**2 + (tree_bot - monkey_bot)**2)
        # dist_to_tree = np.array([dist_to_tree_bot, dist_to_tree_top])

        #
        # Transformations
        #

        # Bin all continuous pixel ranges into groups of 50 (e.g. 400-449 -> 4)
        dist_to_tree_center_trans = np.floor_divide(dist_to_tree_center, 25)
        if dist_to_tree_center_trans < 0:
            dist_to_tree_center_trans = 0
        dist_to_ground_trans = np.floor_divide(dist_to_ground, 25)
        dist_to_tree = np.floor_divide(tree_dist, 25)
        # Take gravity effects on monkey velocity into account
        vel_trans = monkey_vel - (self.gravity if self.gravity is not None else 0)
        # Bin velocity into groups of 10
        motion_trans = int(vel_trans/10)
        new_state = (above_tree_center, dist_to_ground_trans, dist_to_tree, motion_trans)
        return new_state

    def calc_gravity(self, current_state, last_state):
        last_monkey_vel = last_state['monkey']['vel']
        current_monkey_vel = current_state['monkey']['vel']
        self.gravity = last_monkey_vel - current_monkey_vel

    def q_lookup(self, s, a):

        # if (s, a) not in self.q:
        #     # Return closest Q entry if near bottom of screen
        #     if s[1] <= self.NEAR_GROUND:
        #         return self.q[]
        #     # Return closest Q entry if tree center is within 50
        #    return 0

        # Action
        closest_a = None
        if a in self.q:
            closest_a = a
        else:
            # Not in Q
            return 0

        # Above tree center
        closest_above_tree_center = None
        if s[0] in self.q[a]:
            closest_above_tree_center = s[0]
        else:
            # Not in Q
            if (not s[0]) in self.q[a]:
                closest_above_tree_center = not s[0]
            else:
                return 0

        # Dist to ground
        closest_dist_to_ground = None
        if s[1] in self.q[a][closest_above_tree_center]:
            closest_dist_to_ground = s[1]
        else:
            # Not in Q
            # Find closest dist to ground
            for dist, q_val in self.q[a][closest_above_tree_center].items():
                if (closest_dist_to_ground is None or np.abs(closest_dist_to_ground - s[2]) > np.abs(int(dist) - s[3])):
                    closest_dist_to_ground = int(dist)
            if closest_dist_to_ground is None:
                return 0

        # Dist to tree
        closest_dist_to_tree = None
        if s[2] in self.q[a][closest_above_tree_center][closest_dist_to_ground]:
            closest_dist_to_tree = s[2]
        else:
            # Not in Q
            # Find closest dist to tree
            for dist, q_val in self.q[a][closest_above_tree_center][closest_dist_to_ground].items():
                if closest_dist_to_tree is None or np.abs(closest_dist_to_tree - s[2]) > np.abs(int(dist) - s[3]):
                    closest_dist_to_tree = int(dist)
            if closest_dist_to_tree is None:
                return 0

        # Motion
        closest_motion = None
        if s[3] in self.q[a][closest_above_tree_center][closest_dist_to_ground][closest_dist_to_tree]:
            closest_motion = s[3]
        else:
            # Not in Q
            # Find closest motion
            for motion, q_val in self.q[a][closest_above_tree_center][closest_dist_to_ground][closest_dist_to_tree].items():
                if closest_motion is None or np.abs(closest_motion - s[3]) > np.abs(int(motion) - s[3]):
                    closest_motion = int(motion)
            if closest_motion is None:
                return 0

        return self.q[a][closest_above_tree_center][closest_dist_to_ground][closest_dist_to_tree][closest_motion]

    def q_update(self, s, a, r, s_prime):
        try:
            # print(
            #     "Already seen (" + str(s) + ", " + str(a) + ") = " +
            #     str(self.q[a][s[0]][s[1]][s[2]][s[3]]))
            # Q-learning algorithm

            max_q = np.max(
                [self.q_lookup(s_prime, 0), self.q_lookup(s_prime, 1)])
            self.q[a][s[0]][s[1]][s[2]][s[3]] = (
                1 - self.LEARNING_RATE) * self.q_lookup(
                s, a) + self.LEARNING_RATE * (r + self.DISCOUNT * max_q)
        except:
            # Haven't seen this state/action pair before in Q
            if a not in self.q:
                self.q[a] = {}
            if s[0] not in self.q[a]:
                self.q[a][s[0]] = {}
            if s[1] not in self.q[a][s[0]]:
                self.q[a][s[0]][s[1]] = {}
            if s[2] not in self.q[a][s[0]][s[1]]:
                self.q[a][s[0]][s[1]][s[2]] = {}
            if s[3] not in self.q[a][s[0]][s[1]][s[2]]:
                self.q[a][s[0]][s[1]][s[2]][s[3]] = 0
            # print(
            #     "New state (" + str(s) + ", " + str(a) + ") = " +
            #     str(self.q[a][s[0]][s[1]][s[2]][s[3]]))

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here
        # based on the current state and the last state.
        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.
        # print("action call", state)


        # At state s (self.last_state), we took action a (self.last_action), got reward r (self.last_reward), and now are in state s' and need to decide a'
        s_prime = self.discrete_state(state)

        if self.last_action is None:
            # first action, default to not doing anything to measure gravity
            a_prime = 0

        else:
            # Calculate gravity based on monkey position
            if self.gravity is None:
                self.calc_gravity(state, self.last_state_full)
            # This new action a' will not be the very first action we take
            # Thus, we can safely check previous state/action (s/a)
            a = self.last_action
            s = self.last_state_discrete
            r = self.last_reward
            self.q_update(s, a, r, s_prime)

            swing = self.q_lookup(s_prime, 0)
            jump = self.q_lookup(s_prime, 1)

            a_prime = np.argmax([swing, jump])

            # print(
            #     "(s, a, r, s', a'): " + str(s) + ", " + str(a) + ", " +
            #     str(r) + ", " + str(s_prime) + ", " + str(a_prime))
            # print(
            #     str(self.q_lookup(s_prime, 0)) + " v. " +
            #     str(self.q_lookup(s_prime, 1)))

            # Get new policy from Q
            a_prime = self.greedy_epsilon([swing, jump])

        # Save current s_prime/a_prime as last s/a
        self.last_action = a_prime
        self.last_state_discrete = s_prime
        self.last_state_full = state
        return a_prime

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


class DhruvQLearner(object):

    def __init__(self, discount=0.90, epsilon=0.01):
        # Constants - Same each game
        self.BINS = 50
        self.VBINS = 5

        self.DISCOUNT = discount
        self.EPSILON = epsilon
        self.SCREEN_WIDTH = 600
        self.SCREEN_HEIGHT = 400
        # self.gravity = None

        self.Q = np.zeros(
            (2, int(self.SCREEN_WIDTH / self.BINS) + 1,
             int(self.SCREEN_HEIGHT / self.BINS) + 1, self.VBINS))
        self.k = np.zeros(
            (2, int(self.SCREEN_WIDTH / self.BINS) + 1,
             int(self.SCREEN_HEIGHT / self.BINS) + 1, self.VBINS))

        self.iters = 0
        self.mem = [0, 0]
        self.epoch = 1

        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here
        # based on the current state and the last state.
        # You'll need to select an action and return it.
        # Return 0 to swing and 1 to jump.

        self.iters += 1

        tree_dist = int(state['tree']['dist'] / self.BINS)

        if tree_dist < 0:
            tree_dist = 0

        tree_bot = int(
            (state['tree']['top'] - state['monkey']['top'] + 0) / self.BINS)

        monkey_vel = int((state['monkey']['vel']) / 20)

        if np.abs(monkey_vel) > 2:
            monkey_vel = np.sign(monkey_vel) * 2
        if npr.rand() < 0.5:
            aprime = 0
        else:
            aprime = 1

        if not self.last_action == None:

            last_tree_dist = int(self.last_state['tree']['dist'] / self.BINS)
            last_tree_bot = int(
                (self.last_state['tree']['top'] -
                    self.last_state['monkey']['top']) / self.BINS)
            last_monkey_vel = int(self.last_state['monkey']['vel'] / 20)

            if np.abs(last_monkey_vel) > 2:
                last_monkey_vel = np.sign(last_monkey_vel) * 2

            qmax = np.max(self.Q[:, tree_dist, tree_bot, monkey_vel])

            if self.Q[1][tree_dist, tree_bot, monkey_vel] > self.Q[0][tree_dist, tree_bot, monkey_vel]:
                aprime = 1
            else:
                aprime = 0

            if self.k[aprime][tree_dist, tree_bot, monkey_vel] > 0:
                # reduce epsilon if you have a high Q reward
                eps = self.EPSILON / self.k[aprime][tree_dist, tree_bot, monkey_vel]
            else:
                eps = self.EPSILON

            r = npr.rand()

            if r < eps:
                if r < 0.5:
                    aprime = 0
                else:
                    aprime = 1

            alph = 1 / self.k[self.last_action][last_tree_dist, last_tree_bot, last_monkey_vel] # dynamically set learning rate
            self.Q[self.last_action][last_tree_dist, last_tree_bot, last_monkey_vel] += alph*(self.last_reward + self.DISCOUNT * qmax - self.Q[self.last_action][last_tree_dist, last_tree_bot, last_monkey_vel])

        self.mem[0] = state['monkey']['top']

        self.last_action = aprime
        self.last_state = state
        self.k[aprime][tree_dist, tree_bot, monkey_vel] += 1  # ERROR HERE
        return aprime

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):

    '''
    Driver function to simulate learning by
    having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(  # Don't play sounds.
            sound=False,
            text="Epoch %d" % (ii),  # Display the epoch on screen.
            tick_length=t_len,          # Make game ticks super fast.
            action_callback=learner.action_callback,
            reward_callback=learner.reward_callback)
        # Loop until you hit something.
        while swing.game_loop():
            pass
        # Save score history.
        hist.append(swing.score)
        # print(swing.score)
        # Reset the state of the learner.
        # print("---- Q ----")
        # print(learner.Q)
        # print("\n\n")
        # print("---- NEW GAME ----")
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

    ITERS = 500
    plt.ylabel("Scores")
    plt.xlabel("Epochs")
    colors = ["r", "g", "b", "m", "y"]

    # plt.title("Testing Various Epsilon Values, discount=0.9")
    # epsilon_to_test = [0.01, 0.03, 0.05, 0.07, 0.09]  # , 0.1]

    # for idx, epsilon in enumerate(epsilon_to_test):
    #     # Select agent.
    #     agent = DhruvQLearner(epsilon=epsilon)

    #     # Empty list to save history.
    #     hist = []

    #     # Run games.
    #     run_games(agent, hist, ITERS, 1)

    #     # Save history.
    #     print "epsilon", epsilon, hist
    #     np.save('hist-epsilon-%f' % epsilon, np.array(hist))
    #     plt.scatter(range(ITERS), hist, label="%f" % epsilon, marker=".", color=colors[idx])

    # # appending pre-generated data from epsilon = 0.1
    # hist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0, 2, 0, 1, 2, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 3, 0, 1, 0, 2, 0, 0, 0, 0, 2, 1, 1, 0, 3, 0, 1, 1, 1, 3, 1, 2, 0, 2, 0, 0, 2, 1, 0, 4, 1, 0, 2, 5, 0, 0, 0, 0, 0, 2, 0, 2, 1, 1, 1, 0, 0, 0, 4, 0, 1, 2, 1, 1, 0, 3, 0, 3, 0, 5, 3, 1, 5, 1, 0, 6, 5, 0, 4, 0, 1, 0, 0, 2, 0, 2, 4, 11, 2, 0, 1, 0, 11, 0, 0, 9, 7, 7, 1, 2, 0, 0, 1, 1, 7, 1, 1, 0, 11, 0, 23, 0, 5, 4, 1, 4, 0, 1, 0, 27, 1, 0, 1, 1, 1, 0, 0, 0, 3, 6, 15, 0, 4, 0, 14, 0, 10, 8, 0, 0, 1, 42, 0, 0, 0, 16, 0, 1, 10, 0, 10, 16, 0, 0, 1, 9, 1, 8, 0, 14, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 10, 8, 3, 8, 3, 0, 0, 1, 5, 8, 0, 0, 3, 21, 1, 0, 0, 11, 0, 0, 7, 1, 2, 0, 0, 0, 0, 53, 1, 3, 2, 0, 0, 3, 3, 1, 0, 12, 17, 1, 60, 4, 2, 0, 28, 3, 0, 0, 3, 2, 0, 0, 0, 1, 0, 117, 0, 4, 17, 1, 55, 1, 3, 0, 0, 0, 4, 0, 2, 21, 19, 0, 36, 74, 149, 105, 6, 137, 2, 0, 8, 10, 0, 40, 3, 0, 39, 5, 66, 4, 88, 1, 0, 45, 85, 3, 2, 0, 0, 5, 0, 18, 0, 0, 0, 2, 75, 31, 1, 60, 1, 1, 33, 0, 50, 0, 0, 3, 0, 1, 0, 0, 290, 43, 5, 28, 0, 0, 0, 2, 8, 6, 56, 0, 165, 0, 0, 0, 0, 7, 0, 4, 10, 129, 5, 1, 1, 0, 26, 0, 7, 0, 1, 0, 28, 1, 0, 1, 1, 14, 116, 0, 2, 0, 2, 0, 1, 0, 68, 0, 0, 81, 0, 0, 0, 1, 1, 1, 1, 0, 25, 1, 0, 90, 0, 0, 168, 0, 1, 70, 40, 5, 2, 2, 3, 0, 0, 0, 15, 0, 85, 0, 28, 0, 54, 1, 0, 0, 74, 106, 0, 0, 163, 1, 1, 127, 3, 3, 46, 214, 13, 15, 89, 2, 22, 54, 0, 2, 0, 7, 186, 31, 1, 60, 3, 5, 1, 72, 5, 101, 7, 1, 3, 0, 8, 5, 29, 4, 0, 32, 0, 5, 3, 2, 8, 3, 1, 1, 6, 0, 0, 20, 1, 0, 13, 82, 28, 15, 0, 2, 1, 47, 0, 1, 1, 132, 7, 1, 93, 0, 1, 25, 26, 147, 56, 42, 0, 2, 57, 7, 35, 0, 1, 4, 0, 2, 13, 7, 1, 154, 0, 5, 55, 0, 150, 0, 49, 2, 0, 0, 0, 1, 8, 1, 1, 6, 118, 82, 19, 0, 36, 0, 1, 3, 0, 37, 0, 23, 0, 52, 0, 0, 41, 10, 0, 28, 0, 1, 0, 8, 0, 0, 117, 17, 35, 0, 44, 7, 29, 3, 1, 0, 57, 3, 0, 219, 30, 0, 3, 0, 135, 0, 21, 24, 26, 103, 0, 4, 51, 38, 2, 56, 3, 42, 0, 48, 2, 0, 0, 1, 0, 8, 19, 72, 1, 2, 2, 36, 1, 66, 31, 0, 1, 42, 1, 4, 45, 0, 13, 44, 1, 28, 33, 0, 78, 2, 22, 24, 1, 41, 2, 0, 0, 82, 1, 1, 0, 94, 1, 0, 20, 68, 68, 1, 119, 1, 14, 2, 21, 10, 4, 0, 0, 8, 1, 1, 0, 2, 0, 37, 27, 18, 1, 66, 0, 2, 3, 56, 1, 1, 11, 0, 85, 0, 2, 1, 190, 2, 8, 31, 121, 1, 0, 3, 48, 81, 2, 2, 0, 0, 55, 102, 48, 5, 2, 0, 0, 57, 7, 0, 1, 12, 1, 31, 99, 3, 0, 413, 27, 38, 148, 4, 0, 128, 5, 11, 29, 2, 0, 27, 1, 0, 135, 50, 1, 2, 52, 106, 374, 3, 0, 7, 1, 0, 109, 0, 0, 0, 5, 0, 355, 0, 82, 7, 44, 67, 1, 160, 0, 3, 0, 15, 3, 0, 18, 73, 241, 270, 0, 0, 1, 28, 2, 0, 1, 0, 11, 46, 0, 103, 0, 213, 51, 1, 0, 0, 0, 17, 30, 0, 1, 1, 73, 36, 79, 149, 84, 23, 50, 0, 112, 70, 1, 4, 3, 167, 0, 0, 0, 0, 4, 0, 0, 27, 2, 0, 0, 172, 207, 0, 3, 50, 0, 1, 22, 1, 260, 38, 1, 0, 209, 0, 1, 1, 0, 68, 18, 14, 0, 0, 81, 0, 5, 161, 6, 0, 34, 1, 1, 0, 1, 4, 27, 184, 2, 24, 0, 6, 1, 94, 30, 86, 6, 0, 0, 0, 66, 3, 54, 0, 2, 99, 457, 54, 63, 0, 4, 0, 20, 41, 0, 1, 0, 0, 5, 19, 234, 76, 1, 165, 4, 28, 1, 24, 0, 74, 72, 5, 34, 1, 0, 18, 1, 0, 31, 0, 142, 196, 2, 1, 0, 0, 0, 71, 148, 1, 117, 1, 96, 0, 1, 189, 89, 194, 0, 0, 21, 33, 45, 1, 1, 0, 83, 77, 221, 65, 3, 15, 254, 206, 28, 19, 108]
    # plt.scatter(range(ITERS), hist[:ITERS], label="0.1", marker=".", color=color[-1])

    # plt.legend(loc='upper left')
    # plt.show()
    # plt.savefig("epsilon-change.png")

    plt.title("Testing Various Discount Values, epsilon=0.07")
    discount_to_test = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for idx, discount in enumerate(discount_to_test):
        # Select agent.
        agent = DhruvQLearner(discount=discount, epsilon=0.7)

        # Empty list to save history.
        hist = []

        # Run games.
        run_games(agent, hist, ITERS, 1)
        print "discount", discount, hist
        # Save history.
        np.save('hist-discount-%f' % discount, np.array(hist))
        plt.scatter(range(ITERS), hist, label="%f" % discount,
                    color=colors[idx])

    plt.legend(loc='upper left')
    plt.show()
    plt.savefig("discount-change.png")
