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



class dqnhfoAgent(object):
  """ This class holds the basic state info for the agent, and supports
      the API methods for controlling HFO and getting information """

  def __init__(self, server_port, image_port, teamplayer=False):
    self.reward = 0
    self.screen = None
    self.terminal = False
    self.status = 0
    self.actions = []
    self.last_action = NOOP
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
    self.img_socket.settimeout(1)
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
    self.last_action = action
    if PASS == action:
        ## Tweak for two players - 11 and 7
        teammate_unum = 18 - self.hfo_env.playerOnBall().unum
        self.hfo_env.act(action, teammate_unum)
        return
    self.hfo_env.act(action)


class shooterAgent(dqnhfoAgent):
    def getActions(self):
        self.actions = [MOVE, SHOOT, DRIBBLE, PASS, MOVE_TO]
        return lua.eval("{{{}, {}, {}, {}, {}}}".format(*self.actions))

    def act(self, action):
        if MOVE_TO == action:
            self.last_action = MOVE_TO
            x_loc = uniform(13.0, 26.0)
            y_loc = uniform(-20.0, 20.0)
            self.hfo_env.act(action, x_loc, y_loc)
        else:
            dqnhfoAgent.act(self, action)

    def getReward(self):
        features = self.hfo_env.getState()
        self_x = features[0]
        self_y = features[1]
        teammate_x = features[13]
        teammate_y = features[14]
        base_reward = dqnhfoAgent.getReward(self)
        if features[5]:
            # if kickable
            if SHOOT == self.last_action:
               self.reward = base_reward + 10
            elif PASS == self.last_action:
               self.reward = base_reward - 100
        elif (MOVE_TO == self.last_action and
              13.0 < self_x and 26.0 > self_x and
              -20.0 < self_y and 20.0 > self_y):
            # Player is in good area to kick from
            self.reward = base_reward + 2

        # Set negative reward for players close to each other
        sqdist = pow(teammate_x - self_x, 2) + pow(teammate_y - self_y, 2)
        if sqdist < 225:
            distreward = min(2.0/sqdist, 10)
            self.reward -= distreward

        return self.reward

class midfielderAgent(dqnhfoAgent):
    def getActions(self):
        self.actions = [MOVE, SHOOT, DRIBBLE, PASS, MOVE_TO]
        return lua.eval("{{{}, {}, {}, {}, {}}}".format(*self.actions))

    def act(self, action):
        if MOVE_TO == action:
            self.last_action = MOVE_TO
            x_loc = uniform(13.0, 26.0)
            y_loc = uniform(-20.0, 20.0)
            self.hfo_env.act(action, x_loc, y_loc)
        else:
            dqnhfoAgent.act(self, action)

    def getReward(self):
        features = self.hfo_env.getState()
        self_x = features[0]
        self_y = features[1]
        teammate_x = features[13]
        teammate_y = features[14]
        base_reward = dqnhfoAgent.getReward(self)
        if features[5]:
            # if kickable
            if SHOOT == self.last_action:
               self.reward = base_reward - 100
            elif PASS == self.last_action:
               self.reward = base_reward + 10

        # Set negative reward for players close to each other
        sqdist = pow(teammate_x - self_x, 2) + pow(teammate_y - self_y, 2)
        if sqdist < 225:
            distreward = min(2.0/sqdist, 10)
            self.reward -= distreward

        return self.reward
