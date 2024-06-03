import os
import subprocess
import signal
import sys


def command_result(cmd):
    return os.popen(cmd).read().strip()


def execute_command(cmd):
    process = subprocess.Popen(cmd, shell=True)
    try:
        process.wait()
        exit_code = process.returncode
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        sys.exit(process.returncode)

    return exit_code
