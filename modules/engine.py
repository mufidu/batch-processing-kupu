from btreff import BtrflyNet
from PIL import Image
from torchvision import transforms
from os.path import exists
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from natsort import natsorted
import concurrent.futures
import argparse
import shutil
from tqdm import tqdm

matplotlib.use('Agg')

idx = 0
tqdm.monitor_interval = 0 

def render_data(src_front, src_back, dst_front, dst_back, img_front, img_back, model, num_files, index=0):
    global idx
    idx += 1
    # print(f"Rendering {img_front}... ({idx}/{num_files})")

    tfms = transforms.Compose([transforms.ToTensor()])

    X, y = [], []

    X.append(
        [
            tfms(Image.open(f"{src_front}/{img_front}")).tolist(),
            tfms(Image.open(f"{src_back}/{img_back}")).tolist(),
        ]
    )
    y.append(
        [
        ]
    )
    X_valid, y_valid = torch.Tensor(X), torch.Tensor(y)
    n_data_valid = len(X_valid)

    # def masking_torch(msk):
    #     return torch.Tensor(
    #         [(msk == i).cpu().numpy().astype("float") for i in range(13)]
    #     )

    # More efficient masking_torch
    def masking_torch(msk):
        mask_list = [(msk == i).cpu().numpy().astype("float") for i in range(13)]
        mask = np.array(mask_list)
        return torch.Tensor(mask)

    inp_size = (1, 3, 512, 128)
    dsc_size = (1, 1, 13, 512, 128)

    for i in range(n_data_valid):
        x, y = X_valid[i], y_valid[i]

        out_ant, out_pos = model(x[0].reshape(inp_size), x[1].reshape(inp_size))

        tmp = torch.cat(
            [
                masking_torch(out_ant.argmax(axis=1)[0]).reshape(dsc_size),
                masking_torch(out_pos.argmax(axis=1)[0]).reshape(dsc_size),
            ],
            axis=1,
        )

        if i == 0:
            y_predv = tmp + 0
        else:
            y_predv = torch.cat([y_predv, tmp], axis=0)

    # RGB
    cp = {
        0: [1.0, 1.0, 1.0],
        1: [0.6901961, 0.9019608, 0.05098039],
        2: [0.0, 0.5921569, 0.85882354],
        3: [0.49411765, 0.9019608, 0.8862745],
        4: [0.6509804, 0.21568628, 0.654902],
        5: [0.9019608, 0.6156863, 0.7058824],
        6: [0.654902, 0.43137255, 0.3019608],
        7: [0.47843137, 0.0, 0.09411765],
        8: [0.22352941, 0.25490198, 0.72156864],
        9: [0.9019608, 0.85490197, 0.0],
        10: [0.9019608, 0.44705883, 0.13725491],
        11: [0.05098039, 0.7372549, 0.24313726],
        12: [0.9019608, 0.7137255, 0.08627451],
    }

    # Color mapping
    def map_clr(mask):
        res = []
        for row in mask:
            new_row = [cp[x] for x in row]
            res.append(new_row)
        return np.array(res)

    n = 4
    np.random.seed(76)
    vl_idx = np.random.choice(range(n_data_valid), n)

    # img size
    # plt.figure(figsize=(2, 5))
    fig = plt.figure(frameon=False)
    fig.set_figwidth(2)
    fig.set_figheight(5)

    # Disable bg
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Front
    # plt.subplot(2, n*2, (2*i)+1)
    # ax.imshow(X_valid[vl_idx[i]][0].permute(1, 2, 0))
    ax.imshow(map_clr(y_predv[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)
    plt.savefig(f"{dst_front}/{img_front}")

    # Back
    # plt.subplot(2, n*2, (2*i)+2)
    # ax.imshow(X_valid[vl_idx[i]][1].permute(1, 2, 0))
    ax.imshow(map_clr(y_predv[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)
    plt.savefig(f"{dst_back}/{img_back}")

    # Delete the figure after saving to save memory
    plt.close(fig)

def main(src_front, src_back, threads, max_threads=4):
    print("=================================================")
    print("INFERENCING STARTED\n")

    # Load the model
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kupu = BtrflyNet().to(device)
    kupu.load_state_dict(
        torch.load("models/model-eff0406a.pt", map_location=torch.device(device))
    )
    print("Model loaded\n")

    # Get the files
    src_front = f"{src_front}"
    src_back = f"{src_back}"
    files_front = natsorted(os.listdir(src_front))
    files_back = natsorted(os.listdir(src_back))

    # If src contains "preprocessed", replace it with "processed", else, append "_processed"
    dst_front = src_front.replace("preprocessed", "processed") if "preprocessed" in src_front else f"{src_front}_processed"
    dst_back = src_back.replace("preprocessed", "processed") if "preprocessed" in src_back else f"{src_back}_processed"

    # Check if dst exists, else, delete the folder and create a new one
    if not exists(dst_front):
        os.makedirs(dst_front)
        print(f"Created {dst_front}")
    else:
        # Delete previous processed images
        shutil.rmtree(dst_front)
        os.makedirs(dst_front)
        printed_dst = dst_front.split("/")[-1]
        print(f"Deleted previous processed images in {printed_dst}")
    if not exists(dst_back):
        os.makedirs(dst_back)
        print(f"Created {dst_back}")
    else:
        # Delete previous processed images
        shutil.rmtree(dst_back)
        os.makedirs(dst_back)
        printed_dst = dst_back.split("/")[-1]
        print(f"Deleted previous processed images in {printed_dst}")

    # Get the number of files
    num_files = len(files_front)

    # Run the inference
    if threads:
        print(f"Program running in threaded mode with {max_threads} threads\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for i, (ff, fb) in enumerate(zip(files_front, files_back)):
                future = executor.submit(render_data, src_front, src_back, dst_front, dst_back, ff, fb, kupu, num_files, i)
                futures.append(future)

            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                # Handle any exceptions that might have occurred during rendering
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}\n")
    else:
        print("Program running in non-threaded mode\n")
        for ff, fb in tqdm(zip(files_front, files_back), total=len(files_front)):
            render_data(src_front, src_back, dst_front, dst_back, ff, fb, kupu, num_files)

    # Write report
    print("\nINFERENCING DONE")
    print("=================================================")
    print(f"\n{num_files} images processed to {dst_front} and {dst_back}")
    with open("logs/report.txt", "w") as f:
        f.write(f"{num_files} images processed to {dst_front} and {dst_back}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the inference script.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads to use.")
    parser.add_argument("--src_front", required=True, help="Path to source directory for front images.")
    parser.add_argument("--src_back", required=True, help="Path to source directory for back images.")
    args = parser.parse_args()

    main(args.src_front, args.src_back, args.threads, max_threads=args.threads if args.threads > 0 and args.threads < 128 else 4)