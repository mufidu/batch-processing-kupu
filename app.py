import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import subprocess
import threading
import queue

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bonescan Segment Batch Processing")

        self.src_front = tk.StringVar()
        self.src_back = tk.StringVar()
        self.use_threads = tk.IntVar()
        self.num_threads = tk.StringVar()

        self.create_widgets()
        self.engine_thread = None
        self.preprocessing_thread = None
        self.output_queue = queue.Queue()

    def create_widgets(self):
        self.root.state('zoomed')  # Maximize the window

        tk.Label(self.root, text="Select Source Folders:").pack(pady=10)

        tk.Button(self.root, text="Select front folder", command=self.browse_src_front).pack()
        tk.Label(self.root, textvariable=self.src_front).pack()

        tk.Button(self.root, text="Select back folder", command=self.browse_src_back).pack()
        tk.Label(self.root, textvariable=self.src_back).pack()

        tk.Label(self.root, text="========").pack()

        tk.Button(self.root, text="Run Preprocessing", command=self.run_preprocessing_threaded).pack(pady=10)
        tk.Button(self.root, text="Run Processing", command=self.run_engine_threaded).pack(pady=10)

        tk.Label(self.root, text="").pack()
        tk.Label(self.root, text="Output:").pack()
        self.output_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, state=tk.DISABLED)
        self.output_text.pack(padx=10, pady=10)
        tk.Button(self.root, text="Clear Output", command=self.clear_output).pack()

    def browse_src_front(self):
        folder = filedialog.askdirectory()
        if folder:
            self.src_front.set(folder)

    def browse_src_back(self):
        folder = filedialog.askdirectory()
        if folder:
            self.src_back.set(folder)

    def run_preprocessing_threaded(self):
        if self.preprocessing_thread is None or not self.preprocessing_thread.is_alive():
            self.clear_output()  # Clear output before running preprocessing

            def run_preprocessing():
                src_front = self.src_front.get()
                src_back = self.src_back.get()

                if not src_front or not src_back:
                    messagebox.showerror("Error", "Source folders are not selected.")
                    return

                try:
                    cmd = ["venv/Scripts/python.exe", "modules/preprocessing.py", "--src_front", src_front, "--src_back", src_back]
                    self.run_command(cmd)
                except Exception as e:
                    self.append_output(f"Preprocessing error: {str(e)}")

            self.preprocessing_thread = threading.Thread(target=run_preprocessing)
            self.preprocessing_thread.start()

    def run_engine_threaded(self):
        if self.engine_thread is None or not self.engine_thread.is_alive():
            self.clear_output()  # Clear output before running engine

            def run_engine():
                src_front = self.src_front.get()
                src_back = self.src_back.get()

                if not src_front or not src_back:
                    messagebox.showerror("Error", "Source folders are not selected.")
                    return

                use_threads = self.use_threads.get()
                num_threads = self.num_threads.get()

                try:
                    src_front_processing = f"{src_front}_preprocessed_accepted"
                    src_back_processing = f"{src_back}_preprocessed_accepted"
                    cmd = ["venv/Scripts/python.exe", "modules/engine.py", "--src_front", src_front_processing, "--src_back", src_back_processing]
                    if use_threads:
                        cmd.extend(["--threads", num_threads])
                    self.run_command(cmd)

                except Exception as e:
                    if str(e) == "argument --threads: invalid int value: ''":
                        messagebox.showerror("Error", "Please enter a valid number in the threads field.")
                    elif str(e) == "RuntimeError:":
                        messagebox.showerror("Error", "The images are not suitable for processing.")
                    else:
                        self.append_output(f"Engine error: {str(e)}")

            self.engine_thread = threading.Thread(target=run_engine)
            self.engine_thread.start()

    def run_command(self, cmd):
        try:
            # Make the process display the output in real time
            process = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
            for line in iter(process.stdout.readline, ""):
                self.append_output(line)
                print(line.strip())
            process.wait()
        except Exception as e:
            self.append_output(f"Error: {str(e)}")

    def append_output(self, text):
        self.output_queue.put(text)

    def update_gui(self):
        while True:
            try:
                line = self.output_queue.get(block=True, timeout=0.001)
            except queue.Empty:
                pass  # Continue waiting for new output
            else:
                self.output_text.config(state=tk.NORMAL)
                self.output_text.insert(tk.END, line)
                self.output_text.see(tk.END)  # Scroll to the end of the text
                self.output_text.config(state=tk.DISABLED)
                self.output_text.update()  # Update the GUI
    
    def start_gui_update_thread(self):
        thread = threading.Thread(target=self.update_gui)
        thread.daemon = True
        thread.start()

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        self.output_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    app.start_gui_update_thread()
    root.mainloop()
