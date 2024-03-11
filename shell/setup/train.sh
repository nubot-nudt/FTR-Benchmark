ptask-train() {
    if [ "$#" -ne 0 ]; then
        cd ${PTASK_HOME}
        bash ${PTASK_SHELL}/train/$@
    fi

}

_ptask_train_completion() {
    local cur=${COMP_WORDS[COMP_CWORD]}
    words=$(ls -1 ${PTASK_SHELL}/train)
    COMPREPLY=( $(compgen -W "$words" -- $cur) )
}

complete -F _ptask_train_completion ptask-train