import sys
import subprocess

from math import ceil
import os
import shutil

fac_name = sys.argv[1]
number = int(sys.argv[2])
max_process = 10
if len(sys.argv) > 3:
    max_process = int(sys.argv[3])

for i in range(ceil(number / max_process)):
    cmd = ""
    for j in range(max_process):
        if i*max_process+j >= number:
            break
        if cmd != "":
            cmd += " & "
        urdf_path = f"outputs/{fac_name}/{i*max_process+j}/scene.urdf"
        if os.path.exists(urdf_path):
            pass
        elif os.path.exists(f"outputs/{fac_name}/{i*max_process+j}"):
            shutil.rmtree(f"outputs/{fac_name}/{i*max_process+j}")
        cmd += f"python -m infinigen_examples.generate_individual_assets --output_folder outputs/{fac_name} -f {fac_name} -n 1 --seed {i*max_process+j}"
    cmd += " & wait"
    #print(cmd)
    subprocess.call(cmd, shell=True)
        
# subprocess.call(
#     f"python -m infinigen_examples.generate_individual_assets --output_folder outputs/ -f OfficeChairFactory -n 1 --start_seed {}".split(" ")
# )
