# Automatic ANT-POST Bone Scan Image Segmentation

This project provides a tool to preprocess and process medical images of bonescans using Python scripts. It includes functionalities for cropping, resizing, enhancing, and segmenting bone images using the BtrflyNet model.

## Usage

### GUI Application

1. Run the GUI application.

    ```bash
    python app.py
    ```

    Or simply double-click on the `app.exe` file.

2. Select source folders and choose options for preprocessing and processing tasks. You can use imgs/wholeBodyANT and imgs/wholeBodyPOST as sample input folders.

3. Use the GUI interface to run the desired tasks.

4. The processed images will be saved in the output folder, which is created in the same directory as the source folder.

## Development

Before running the scripts, you need to set up a virtual environment and install the required dependencies. You can do this using the following steps:

1. Install `virtualenv` (if not already installed):

    ```bash
    pip install virtualenv
    ```

2. Create a virtual environment:

    ```bash
    virtualenv venv
    ```

3. Activate the virtual environment:
    
    - On Windows:
    
        ```bash
        venv\Scripts\activate
        ```

    - On Linux/macOS:
    
        ```bash
        source venv/bin/activate
        ```

4. Install the required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```

5. Download the model [here](https://telkomuniversityofficial-my.sharepoint.com/:u:/g/personal/mufidu_student_telkomuniversity_ac_id/ETPjRLyY_AxMtOCZcxgs7vUBL1jttRaP_dQ_T2iQwmV0Eg?e=v2GJcb) and place it in this directory.

6. Run the GUI:

    ```bash
    python app.py
    ```

7. Or run the individual scripts:

    ```bash
    # Use provided imgs/wholeBodyANT and imgs/wholeBodyPOST as sample input folders
    # Preprocessing
    python modules/preprocessing.py --src_front imgs/wholeBodyANT --src_back imgs/wholeBodyPOST
    ```
    
    ```bash
    # Processing
    python modules/engine.py \
    --src_front imgs/wholeBodyANT_preprocessed \
    --src_back imgs/wholeBodyPOST_preprocessed
    ```

    Run the scripts with the `--help` flag to see the available options.

## Benchmarking

To benchmark the threaded and nonthreaded versions of the preprocessing and processing scripts, run the following command:

```bash
python modules/benchmarking.py
```

The results will be saved in the `logs/benchmarking.txt` file.

## Building 

```bash
pyinstaller --onefile \
--distpath . \
app.py
```

This will create an executable file (`app.exe`) in the current directory. You can run this file to launch the GUI application.

## Troubleshooting

If you encounter any issues or errors while running the scripts, please ensure that the virtual environment is activated and the required dependencies are installed.
