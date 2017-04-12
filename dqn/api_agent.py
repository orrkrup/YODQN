#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

""" This is an HFO agent designed as an API for the DQN algorithm to play HFO.
    Created by Orr Krupnik, Yona Cohen """

import socket
import sys
from datetime import datetime, timedelta
import lua
from random import uniform
from math import sqrt

sys.path.insert(0, '/home/deep3/HFO/hfo/')
from hfo import *

def runAgent(q, type, server_port, image_port):
    if "shooter" in type:
        print("launching shooter agent...")
        q.put(shooterAgent(server_port, image_port, True))
        print("shooter agent launched")
    else:
        print("launching midfielder agent")
        q.put(midfielderAgent(server_port, image_port, True))
        print("midfielder agent launched")


def runDoubleAgent(server_port, image_port):
    print "initializing shooter and midfielder"
    from threading import Thread
    from Queue import Queue
    sh_q = Queue()
    mid_q = Queue()
    shooter_th = Thread(group=None, target=runAgent, args=(sh_q, 'shooter', server_port, image_port))
    shooter_th.start()
    mid_th = Thread(group=None, target=runAgent, args=(mid_q, 'mid', server_port, image_port))
    mid_th.start()

    #return lua.eval("{{{}, {}}}".format(sh_q.get(), mid_q.get())
    return sh_q.get(), mid_q.get()



class dqnhfoAgent(object):
  """ This class holds the basic state info for the agent, and supports
      the API methods for controlling HFO and getting information """

  def __init__(self, server_port, image_port, teamplayer=False):
    self.reward = 0
    self.max_reward = 0
    self.screen = None
    self.terminal = False
    self.status = 0
    self.actions = []
    self.last_action = NOOP
    self.prev_action = NOOP
    self.team = teamplayer

    print "Initializing api_agent"
    # Create the HFO environment
    self.hfo_env = HFOEnvironment()
    # Connect to the server with the specified
    # feature set. See feature sets in hfo.py/hfo.hpp.
    self.hfo_env.connectToServer(HIGH_LEVEL_FEATURE_SET,
                                 '/home/deep3/HFO/bin/teams/base/config/formations-dt', server_port,
                                 'localhost', 'base_left', False)
    print "api_agent connected to server"

    # image assistance paramteres
    self.wantimage = True
    self.img_buffer = ''
    self.img_recv_time = timedelta(microseconds=1)
    self.num_imgs = 0
    self.height = 252
    self.width = 252

    # Prepare torch in lua space
    lua.execute("require 'torch'")

    # Initialize image receiving port (TCP)
    self.img_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.img_socket.settimeout(5)
    self.img_socket.connect(('localhost', image_port))

    print "api_agent intialization complete"


  def getReward(self):
    # Replace with reward function when we know how to calculate it
    if self.status == IN_GAME:
      self.reward = -0.001
    elif self.status == GOAL:
      self.reward = 1000
    elif self.status == OUT_OF_BOUNDS:
      self.reward = -1
    elif self.status == OUT_OF_TIME:
      self.reward = -1
    elif self.status == CAPTURED_BY_DEFENSE:
      self.reward = -4

    return self.reward


  def agentStep(self):
    terminal = False
    self.status = self.hfo_env.step()
    if not self.status == IN_GAME:
      terminal = True
    return terminal


  def getActions(self):
    ### Change when necessary
    if self.team:
        self.actions = [MOVE, SHOOT, DRIBBLE, PASS]
        return lua.eval("{{{}, {}, {}, {}}}".format(*self.actions))
    else:
        self.actions = [MOVE, SHOOT, DRIBBLE]
        return lua.eval("{{{}, {}, {}}}".format(*self.actions))


  def getStateDims(self):
    return self.height * self.width


  def act(self, action):
    if action == 0:
        action = NOOP
    self.prev_action = self.last_action
    self.last_action = action
    if PASS == action:
        ## Tweak for two players - 11 and 7
        teammate_unum = 18 - self.hfo_env.playerOnBall().unum
        self.hfo_env.act(action, teammate_unum)
        return
    self.hfo_env.act(action)


class shooterAgent(dqnhfoAgent):
    def getActions(self):
        #self.actions = [MOVE, SHOOT, DRIBBLE, PASS, MOVE_TO]
        #return lua.eval("{{{}, {}, {}, {}, {}}}".format(*self.actions))

        ## Sanity check actions
        self.actions = [SHOOT, MOVE_TO, MOVE]
        return lua.eval("{{{}, {}, {}}}".format(*self.actions))


    def act(self, action):
        if MOVE_TO == action:
            self.prev_action = self.last_action
            self.last_action = MOVE_TO
#            x_loc = uniform(0.2, 0.5)
#            y_loc = uniform(-0.4, 0.4)
#            self.hfo_env.act(action, x_loc, y_loc)
            self.hfo_env.act(MOVE_TO, 0.4, 0.4)
        else:
            dqnhfoAgent.act(self, action)

    def getReward(self):
        features = self.hfo_env.getState()
        self_x = features[0]
        self_y = features[1]
        teammate_x = features[13]
        teammate_y = features[14]
        dqnhfoAgent.getReward(self)
        if features[5] > 0:
            # if kickable
            if SHOOT == self.last_action:
               self.reward += 4
            elif PASS == self.last_action:
               self.reward -= 1
        elif MOVE == self.last_action and self.hfo_env.playerOnBall().unum > 0:
            # Negative reward for moving towards ball if there is a player on it
            self.reward -= 1
        # elif (MOVE_TO == self.last_action and
        #       13.0 < self_x and 26.0 > self_x and
        #       -20.0 < self_y and 20.0 > self_y):
        #     # Player is in good area to kick from
        #     self.reward = base_reward + 0.01

        # Negative reward for shooter in left half of field:
        if self_y < 0:
            self.reward -= 1

        # Remove goal reward if the midfielder scored by dribbling to goal
        if self.reward > 900 and self.hfo_env.playerOnBall().unum > 0:
            self.reward -= 1000

        # Set negative reward for players close to each other
        sqdist = pow(teammate_x - self_x, 2) + pow(teammate_y - self_y, 2)
        if sqdist < 0.01:
            distreward = min(0.01/sqdist, 2)
            self.reward -= distreward

        ### sanity check - cancel reward shaping
        #self.reward = base_reward
        #print("Shooter: XPOS: {}, YPOS: {}, GAMESTATE: {}, LAST_ACTION: {}, REWARD: {}".format(
        #      self_x, self_y, self.status, self.last_action, self.reward))
        #print("features[5]: " + str(features[5]))
        #if self.reward > 900 and self.status != GOAL:
        #    print ("High reward but not goal")
        #    print("Shooter: XPOS: {}, YPOS: {}, GAMESTATE: {}, LAST_ACTION: {}, REWARD: {}".format(
        #          self_x, self_y, self.status, self.last_action, self.reward))
        return self.reward


class midfielderAgent(dqnhfoAgent):
    def getActions(self):
        #self.actions = [MOVE, SHOOT, DRIBBLE, PASS, MOVE_TO]
        #return lua.eval("{{{}, {}, {}, {}, {}}}".format(*self.actions))

        ### Sanity check actions
        self.actions = [MOVE, DRIBBLE, PASS]
        return lua.eval("{{{}, {}, {}}}".format(*self.actions))

    def act(self, action):
        if MOVE_TO == action:
            self.prev_action = self.last_action
            self.last_action = MOVE_TO
            x_loc = uniform(0.2, 0.5)
            y_loc = uniform(-0.4, 0.4)
            self.hfo_env.act(action, x_loc, y_loc)
        else:
            dqnhfoAgent.act(self, action)

    def getReward(self):
        features = self.hfo_env.getState()
        self_x = features[0]
        self_y = features[1]
        teammate_x = features[13]
        teammate_y = features[14]
        dqnhfoAgent.getReward(self)

        if self.reward > 900:
            ## WARNING: this is hardcoded and will need to be changed if goal reward is ever changed
            self.reward -= 1000 # Midfielder is not suppsed to get scoring reward

        if features[5] > 0:

            # if kickable
            if SHOOT == self.last_action:
                self.reward -= 5
            elif PASS == self.last_action and PASS != self.prev_action:
                self.reward += 5 # High midfielder reward for passing
            elif DRIBBLE == self.last_action:
                self.reward -= 0.05
        elif MOVE == self.last_action:
            self.reward += 0.01

        # Set negative reward for players close to each other
        sqdist = pow(teammate_x - self_x, 2) + pow(teammate_y - self_y, 2)
        if sqdist < 0.01:
            distreward = min(0.01/sqdist, 2)
            self.reward -= distreward

        if self.reward > self.max_reward:
            self.max_reward = self.reward
        ### sanity check - cancel reward shaping
        #self.reward = base_reward
        #print("Midfielder: XPOS: {}, YPOS: {}, GAMESTATE: {}, LAST_ACTION: {}, REWARD: {}, MAX REWARD: {}".format(
        #      self_x, self_y, self.status, self.last_action, self.reward, self.max_reward))
        #print("features[5]: " + str(features[5]))
        if self.reward > 5:
            print("Midfielder reward " + str(self.reward) + ", exiting")
            from sys import exit
            exit(1)
        return self.reward
