import re
import sys
import paramiko
import subprocess
from time import sleep

#GPU_1
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.103.48.134',22,"zhufangyi", "123456")

#GPU_2
ssh2 = paramiko.SSHClient()
ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh2.connect('10.108.211.48',22,"lxx", "lxx")

while True:
    #GPU_1
    stdin, stdout, stderr = ssh.exec_command("nvidia-smi")
    foo = stdout.readlines()
    result  = re.findall(".*W(.*)Default.*",str(foo))
    #GPU_2
    stdin, stdout, stderr = ssh2.exec_command("nvidia-smi")
    foo = stdout.readlines()
    result2 = re.findall(".*W(.*)Default.*",str(foo))

    print("state of GPUs")
    print("         IP                Memory-Usage         Compute M")
    print("10.103.48.134>>>>"+str(result))
    print("10.108.211.48>>>>"+str(result2))
    sleep(2)
    subprocess.call(["clear"])

    


