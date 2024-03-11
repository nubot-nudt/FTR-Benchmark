ptask-experiment() {
    if [ "$#" -ne 0 ]; then
        cd ${PTASK_HOME}
        bash ${PTASK_SHELL}/experiment/$@
    fi

}

_ptask_experiment_completion() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    words=$(ls -1 ${PTASK_SHELL}/experiment)
    COMPREPLY=( $(compgen -W "$words" -- $cur) )
}

complete -F _ptask_experiment_completion ptask-experiment