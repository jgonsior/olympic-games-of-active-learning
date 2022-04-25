#!/bin/bash
N={{BASH_PARALLEL_RUNNERS}}
(
for iteration in {% raw %}{{% endraw %}{{ START }}..{{ END }}{% raw %}}{% endraw %}; do
   ((i=i%N)); ((i++==0)) && wait
   python {{PYTHON_FILE}} --EXP_TITLE {{ EXP_TITLE }} --WORKER_INDEX "$iteration" &
done
)
