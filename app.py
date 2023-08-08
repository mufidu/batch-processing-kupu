import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import subprocess
import threading
import queue

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        self.src_front = tk.StringVar()
        self.src_back = tk.StringVar()
        self.use_threads = tk.IntVar()
        self.num_threads = tk.StringVar()

        self.create_widgets()
        self.engine_thread = None
        self.preprocessing_thread = None

    def create_widgets(self):
        self.root.state('zoomed')  # Maximize the window

        tk.Label(self.root, text="Select Source Folders:").pack(pady=10)

        tk.Button(self.root, text="Select src_front Folder", command=self.browse_src_front).pack()
        tk.Label(self.root, textvariable=self.src_front).pack()

        tk.Button(self.root, text="Select src_back Folder", command=self.browse_src_back).pack()
        tk.Label(self.root, textvariable=self.src_back).pack()

        tk.Checkbutton(self.root, text="Use Threads", variable=self.use_threads).pack()

        tk.Label(self.root, text="Number of Threads:").pack()
        tk.Entry(self.root, textvariable=self.num_threads).pack()

        tk.Button(self.root, text="Run Preprocessing", command=self.run_preprocessing_threaded).pack(pady=10)
        tk.Button(self.root, text="Run Engine", command=self.run_engine_threaded).pack(pady=10)

        tk.Label(self.root, text="").pack()
        tk.Label(self.root, text="Output:").pack()
        self.output_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20, state=tk.DISABLED)
        self.output_text.pack(padx=10, pady=10)
        tk.Button(self.root, text="Clear Output", command=self.clear_output).pack()

        self.output_queue = queue.Queue()
        self.root.after(100, self.process_output_queue)

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
            self.preprocessing_thread = threading.Thread(target=self.run_preprocessing)
            self.preprocessing_thread.start()

    def run_preprocessing(self):
        src_front = self.src_front.get()
        src_back = self.src_back.get()

        if not os.path.exists(src_front) or not os.path.exists(src_back):
            messagebox.showerror("Error", "Source folders do not exist.")
            return

        self.clear_output()  # Clear output before running preprocessing

        try:
            cmd = ["venv/Scripts/python.exe", "preprocessing.py"]
            result = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            for line in result.stdout:
                self.append_output(line)

            result.communicate()  # Ensure subprocess finishes
            result.stdout.close()
        except Exception as e:
            self.append_output(f"Preprocessing error: {str(e)}")

    def run_engine_threaded(self):
        if self.engine_thread is None or not self.engine_thread.is_alive():
            self.engine_thread = threading.Thread(target=self.run_engine)
            self.engine_thread.start()

    def run_engine(self):
        src_front = self.src_front.get()
        src_back = self.src_back.get()

        if not os.path.exists(src_front) or not os.path.exists(src_back):
            messagebox.showerror("Error", "Source folders do not exist.")
            return

        use_threads = self.use_threads.get()
        num_threads = self.num_threads.get()

        self.clear_output()  # Clear output before running engine

        try:
            cmd = ["venv/Scripts/python.exe", "engine.py"]
            if use_threads:
                cmd.extend(["--threads", num_threads])

            result = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            for line in result.stdout:
                self.append_output(line)

            result.communicate()  # Ensure subprocess finishes
            result.stdout.close()
        except Exception as e:
            self.append_output(f"Engine error: {str(e)}")

    def process_output_queue(self):
        try:
            while True:
                output = self.output_queue.get_nowait()
                self.append_output(output)
        except queue.Empty:
            pass

        self.root.after(100, self.process_output_queue)

    def append_output(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)  # Scroll to the end of the text
        self.output_text.config(state=tk.DISABLED)

    def clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        self.output_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
