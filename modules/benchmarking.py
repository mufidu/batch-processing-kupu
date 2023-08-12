import time
import subprocess
import os

python_path = "venv/Scripts/python.exe"

start_time = time.time()
subprocess.run([python_path, "modules/engine.py", "--threads", "4", "--src_front", "imgs/wholeBodyANT_preprocessed_accepted", "--src_back", "imgs/wholeBodyPOST_preprocessed_accepted"])
end_time = time.time()
threaded_time = end_time - start_time

start_time = time.time()
subprocess.run([python_path, "modules/engine.py", "--src_front", "imgs/wholeBodyANT_preprocessed_accepted", "--src_back", "imgs/wholeBodyPOST_preprocessed_accepted"])
end_time = time.time()
non_threaded_time = end_time - start_time

print("=================================================")
# Get number of files in imgs/wholeBodyANT
num_files = len([f for f in os.listdir("imgs/wholeBodyANT") if os.path.isfile(os.path.join("imgs/wholeBodyANT", f))])
print(f"Number of files: {num_files}")
print(f"Elapsed time without threading: {non_threaded_time:.2f} seconds")
print(f"Elapsed time with threading: {threaded_time:.2f} seconds")
speedup = ((non_threaded_time - threaded_time) / non_threaded_time) * 100
print(f"Speedup percentage: {speedup:.2f}%")

# Write to benchmarking.txt
with open("logs/benchmarking.txt", "w") as f:
    f.write("=================================================\n")
    f.write(f"Number of files: {num_files}\n")
    f.write(f"Elapsed time without threading: {non_threaded_time:.2f} seconds\n")
    f.write(f"Elapsed time with threading: {threaded_time:.2f} seconds\n")
    f.write(f"Speedup percentage: {speedup:.2f}%\n")
    f.write("=================================================\n")
