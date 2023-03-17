import os
import subprocess
import select
import signal


def consume(p: subprocess.Popen):
    """
    Consume the standard output and standard error of the specified process.

    Args:
        p (subprocess.Popen): The process to consume the output from.
    """
    for stream in [p.stdout, p.stderr]:
        while True:
            line = stream.readline().decode('utf-8')
            if not line:
                break
            print(line, end='')


def run_command(command: str, cwd=None, env=None, non_block=False):
    """
    Executes a command in the shell and captures its standard output and standard error.

    Args:
        command (str): A string representing the command to execute in the shell.
        cwd (str): A string representing the working directory to execute the command in. Default is None.
        env (dict): A dictionary representing the environment variables to set for the command. Default is None.
        non_block (bool): A flag to indicate whether to run the command in a non-blocking manner. Default is False.

    Returns:
        The return code of the executed command.
    """
    proc = subprocess.Popen(command,
                            cwd=cwd,
                            env=env,
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if non_block:
        return proc

    try:
        while proc.poll() is None:
            # Wait for output from the process
            rlist, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.1)
            for stream in rlist:
                line = stream.readline().decode('utf-8')
                if line:
                    print(line, end='')

        # Read the remaining output
        for stream in [proc.stdout, proc.stderr]:
            while True:
                line = stream.readline().decode('utf-8')
                if not line:
                    break
                print(line, end='')

        # Get the return code of the process
        return_code = proc.wait()

        # Raise an exception if the process returned a non-zero return code
        # if return_code != 0:
        #     raise subprocess.CalledProcessError(return_code, command)
    except KeyboardInterrupt:
        os.kill(proc.pid, signal.SIGINT)

        while proc.poll() is None:
            # Wait for output from the process
            rlist, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.1)
            for stream in rlist:
                line = stream.readline().decode('utf-8')
                if line:
                    print(line, end='')

            # Read the remaining output
        for stream in [proc.stdout, proc.stderr]:
            while True:
                line = stream.readline().decode('utf-8')
                if not line:
                    break
                print(line, end='')

        # Get the return code of the process
        return_code = proc.wait()
    return return_code
