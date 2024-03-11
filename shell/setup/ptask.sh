
export PTASK_HOME="${PTASK_HOME:-$(pwd)}"
export PTASK_HOME=$(realpath ${PTASK_HOME})

export PTASK_SETUP=$PTASK_HOME/shell/setup/ptask.sh

export PTASK_TOOLS=$PTASK_HOME/tools
export PTASK_SCRIPTS=$PTASK_HOME/scripts
export PTASK_SHELL=$PTASK_HOME/shell
export PTASK_SRC=$PTASK_HOME/src
export PTASK_SERVER_HOST="127.0.0.1"

ptask-kill-train() {
  pkill -f "scripts/train_rlgames.py"
  pkill -f "scripts/train_sarl.py"
}

ptask-tensorboard() {
  ptask-home
  tensorboard --logdir runs --bind_all $@
}

isaac-sim() {
  ~/.local/share/ov/pkg/isaac_sim*/isaac-sim.selector.sh $@
}

local-python() {
  python3 $@
}

isaac-python() {
  ~/.local/share/ov/pkg/isaac_sim*/python.sh $@
}

isaac-home() {
  cd ~/.local/share/ov/pkg/isaac_sim*/
}

ptask-home() {
    cd ${PTASK_HOME}
}

ptask-tools() {
    if [ "$#" -ne 0 ]; then
        ptask-home
        local-python ${PTASK_TOOLS}/$@
    fi

}

ptask-shell() {
    if [ "$#" -ne 0 ]; then
        ptask-home
        bash ${PTASK_SHELL}/$@
    fi

}

ptask-scripts() {
    if [ "$#" -ne 0 ]; then
        ptask-home
        isaac-python ${PTASK_SCRIPTS}/$@
    fi

}

_ptask_tools_completion() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    words=$(ls -1 ${PTASK_TOOLS})
    COMPREPLY=( $(compgen -W "$words" -- $cur) )
}

_ptask_shell_completion() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    words=$(ls -1 ${PTASK_SHELL})
    COMPREPLY=( $(compgen -W "$words" -- $cur) )
}

_ptask_scripts_completion() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    words=$(ls -1 ${PTASK_SCRIPTS})
    COMPREPLY=( $(compgen -W "$words" -- $cur) )
}

complete -F _ptask_tools_completion ptask-tools
complete -F _ptask_shell_completion ptask-shell
complete -F _ptask_scripts_completion ptask-scripts

for file in `ls ${PTASK_SHELL}/setup/*.sh | grep -v ptask.sh`
do
    source $file
done
