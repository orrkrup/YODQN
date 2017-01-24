#!/bin/bash

cat $1 | grep "TD" | cut -d"V" -f 2|cut -f2 > ./Results/vavg.txt
cat $1 | grep "reward:" | cut -d "," -f 2 | cut -d " " -f 3 > ./Results/reward.txt
cat $1 | grep "TD" | cut -d"T" -f 2|cut -f2> ./Results/TD.txt
cat $1 | grep "Epoch" | cut -d" " -f3,5 > ./Results/goals.txt 
th ./Results/Plot.lua


