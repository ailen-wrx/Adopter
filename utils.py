import os, subprocess


def run_cmd(command, p=False):
    if p:
        print("Running command:", command)
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p:
        print(proc.stdout)
        print(proc.stderr)
    return


