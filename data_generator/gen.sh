#!/bin/bash  
  
for (( i=0; i<100; i=i+1 )); do
  python NG.py $i;
  echo $i finished;
done