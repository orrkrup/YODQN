#!/bin/bash

datetime=`date +%d%m%y_%H%M`

( ./bin/HFO --no-logging --offense-agents=$1 --defense-npcs=$2 --port=$3  --image_port=$4 --frames-per-trial=1000 &) > DQNLogs/log$datetime

cd ../Orr_Yona_DQN/dqn

sleep 1

if (( $1 == 1 )); then
  echo Running HFO-DQN with one player
  ( ../run_gpu HFO $3 $4 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime'_a'
elif (( $1 == 2 )); then
  if (( $# > 4 )); then
    if [ "$5" == "--onenet"]; then
      echo Runnnin HFO-DQN with two players on one network
      ( ../run_gpu HFO $3 $4 --onenet 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime
    fi
  else
    echo Running HFO-DQN with two players on two networks
    ( ../run_gpu HFO $3 $4 --team 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime'_a'
    sleep 1
    ( ../run_gpu HFO $3 $4 --team 2>&1 | grep -v 'Violated\|kickable\|base_left' &) > Logs/log$datetime'_b'
  fi
fi



cd -
