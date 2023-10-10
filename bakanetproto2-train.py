#!/usr/bin/env python
import os

CFG = "p160"
MAX = 10000
STEP = 1000

for x in range(0, MAX, STEP):
    cmd = f"python bakanetproto2.py --cfg={CFG} -n{STEP}"
    cmd += f" --skip={x} --load"
    x = os.system(cmd)
    if x:
        print(f"Failure during {cmd}")
        quit(x)
    os.system("nvidia-smi")
