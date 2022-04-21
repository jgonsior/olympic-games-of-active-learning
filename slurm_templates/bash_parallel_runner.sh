#!/bin/bash
N={{BASH_PARALLEL_RUNNERS}}
(
for iteration in seq {{START}} {{END}}; do 
   ((i=i%N)); ((i++==0)) && wait
   python {{PYTHON_FILE}} --WORKER_INDEX "$iteration" & 
done
)
