import os
import subprocess
import select
import signal


def run_command(command, cwd=None, env=None):
    """
    Executes a command in the shell and captures its standard output and standard error.

    Args:
        command: a string representing the command to execute in the shell.
        cwd: a string representing the working directory to execute the command in. Default is None.
        env: a dictionary representing the environment variables to set for the command. Default is None.

    Returns:
        The return code of the executed command.
    """
    proc = subprocess.Popen(command,
                            cwd=cwd,
                            env=env,
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
