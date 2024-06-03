#!/bin/bash

source ./shell/setup/ptask.sh

if [ -z "$PTASK_CONDA_NAME" ]; then
  $ISAACSIM_HOME/python.sh $@
  exitcode=$?
else
  source activate $PTASK_CONDA_NAME
  source $ISAACSIM_HOME/setup_conda_env.sh
  python $@
  exitcode=$?
fi
exit $exitcode