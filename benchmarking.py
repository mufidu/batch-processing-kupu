import time
import subprocess

python_path = "venv/Scripts/python.exe"

start_time = time.time()
subprocess.run([python_path, "engine.py"])
end_time = time.time()
non_threaded_time = end_time - start_time

start_time = time.time()
subprocess.run([python_path, "engine.py", "--threading"]) 
end_time = time.time()
threaded_time = end_time - start_time

print("=================================================")
print(f"Elapsed time without threading: {non_threaded_time:.2f} seconds")
print(f"Elapsed time with threading: {threaded_time:.2f} seconds")
speedup = ((non_threaded_time - threaded_time) / non_threaded_time) * 100
print(f"Speedup percentage: {speedup:.2f}%")

# Write to benchmarking.txt
with open("benchmarking.txt", "w") as f:
    f.write("=================================================\n")
    f.write(f"Elapsed time without threading: {non_threaded_time:.2f} seconds\n")
    f.write(f"Elapsed time with threading: {threaded_time:.2f} seconds\n")
    f.write(f"Speedup percentage: {speedup:.2f}%\n")
    f.write("=================================================\n")
