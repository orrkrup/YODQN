#!/bin/bash

# Run HFO-DQN framework.
# Parameters:
# $1: offensive agents
# $2: defensive npcs
# $3: server port
# $4: image port
# $5: --onenet or --load option

datetime=`date +%d%m%y_%H%M`

( ./bin/HFO --offense-on-ball=2 --no-logging --offense-agents=$1 --defense-npcs=$2 --port=$3  --image_port=$4 --frames-per-trial=1000 &) > DQNLogs/log$datetime

cd ../Orr_Yona_DQN/dqn

sleep 1

if (( $1 == 1 )); then
  if (( $# > 4 )); then
    if [ "$5" == "--load" ]; then
      echo Loading old network
      ( ../run_gpu_load HFO $3 $4 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime
    fi
  else
    echo Running HFO-DQN with one player
    ( ../run_gpu HFO $3 $4 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime
  fi
elif (( $1 == 2 )); then
  if (( $# > 4 )); then
    if [ "$5" == "--onenet" ]; then
      echo Running HFO-DQN with two players on one network
      ( ../run_gpu HFO $3 $4 --onenet 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime
    fi
  else
    echo Running HFO-DQN with two players on two networks
    ( ../run_gpu HFOshooter $3 $4 --shooter 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime'_a'
    sleep 1
    ( ../run_gpu HFOmid $3 $4 --midfielder 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime'_b'
  fi
fi



cd -
