#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

""" This is an HFO agent designed as an API for the DQN algorithm to play HFO.
    Created by Orr Krupnik, Yona Cohen """

import socket
import sys
import StringIO
from PIL import Image 
import numpy
#import scipy.misc
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import lua

sys.path.insert(0, '/home/deep3/HFO/hfo/')
from hfo import *
 


class dqnhfoAgent(object):
  """ This class holds the basic state info for the agent, and supports
      the API methods for controlling HFO and getting information """
    
  def __init__(self, server_port, image_port):
    self.reward = 0
    self.screen = None
    self.terminal = False
    self.status = 0
    self.actions = []
    
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
    
    # Initialize image receiving port
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


  def getScreen(self):
    #getstart = datetime.now()
    datacount = self.height * self.width * 3 
    stringimage = ''
    
    while datacount > 0:    
      self.img_buffer += self.img_socket.recv(datacount, 0x40)
      
      buflen = len(self.img_buffer)
      # Stop loop if received nothing:
      if buflen == 0:
        return 

      if datacount - buflen >= 0:
        stringimage += self.img_buffer
        datacount -= buflen
        self.img_buffer = ''
      else: 
        stringimage += self.img_buffer[:datacount]
        self.img_buffer = self.img_buffer[datacount + 1:]
        datacount = 0
    
    arr1d = numpy.frombuffer(stringimage, dtype=numpy.uint8)
    #print len(arr1d)
    arr3d_large = numpy.reshape(arr1d, (self.height, self.width, 3))
    
    ### snippet to print first image array to log and show it 
    #if self.wantimage:
    #  self.wantimage = False
    #  print arr3d
    #  plt.imshow(arr3d)
    #  plt.show()
   
    height = 84
    width = 84 
    arr3d = scipy.misc.imresize(arr3d_large, (height, width, 3))
    
    #getmid = datetime.now()
    
    img_table = lua.eval("{{}, {}, {}}")
    for x in range(width):
      img_table[1][x+1] = lua.eval("{}")
      img_table[2][x+1] = lua.eval("{}")
      img_table[3][x+1] = lua.eval("{}")
      for y in range(height):
        img_table[1][x+1][y+1] = arr3d[x][y][0]/255
        img_table[2][x+1][y+1] = arr3d[x][y][1]/255
        img_table[3][x+1][y+1] = arr3d[x][y][2]/255

    self.screen = img_table 
    #conversion = datetime.now() - getmid
    ### Code to time image receive
    #delta = getmid - getstart
    #self.img_recv_time += delta
    #self.num_imgs += 1
    #print "get screen from tcp (and rescale) was {}".format(delta)
    #print "conversion part was {}".format(conversion)
    return self.screen


  def agentStep(self):
    terminal = False
    self.status = self.hfo_env.step()
    if not self.status == IN_GAME:
      terminal = True
    return terminal


  def getActions(self):
    ### Change when necessary 
    self.actions = [MOVE, SHOOT, DRIBBLE]
    return lua.eval("{{{}, {}, {}}}".format(MOVE, SHOOT, DRIBBLE))
   
 
  def getStateDims(self):
    return self.height * self.width


  def act(self, action):
    if action == 0:
      action = NOOP
    self.hfo_env.act(action) 
    

if __name__ == '__main__':
  ### Create agent object
  agent = dqnhfoAgent()
  
  steps = 0
  while steps < 100:
    # get state from HFO (blocking)
    agent.getScreen() 
    # act in HFO - kick if on ball, move otherwise
    if agent.hfo_env.getState()[5] > 0:
      action = 1
    else:
      action = 0
    agent.act(action)
    # get status from HFO.step
    if agent.agentStep(): 
      steps += 1

  print "Collected {} images.".format(agent.num_imgs)
  print "Average image time: {}".format(agent.img_recv_time / agent.num_imgs)

