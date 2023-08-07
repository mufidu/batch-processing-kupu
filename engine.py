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

matplotlib.use('Agg')

def render_data(src_front, src_back, dst_front, dst_back, img_front, img_back, model, index=0):
    print(f"Rendering {img_front}...")

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

    # ini warna (r,g,b)
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

    # pewarnaan disini
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

    # this for disabling the bg
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # depan
    # plt.subplot(2, n*2, (2*i)+1)
    ax.imshow(X_valid[vl_idx[i]][0].permute(1, 2, 0))
    ax.imshow(map_clr(y_predv[vl_idx[i]][0].argmax(axis=0).numpy()), alpha=0.5)
    plt.savefig(f"{dst_front}/{img_front}")
    print(f"Saved to {dst_front}/{img_front}")

    # belakang
    # plt.subplot(2, n*2, (2*i)+2)
    ax.imshow(X_valid[vl_idx[i]][1].permute(1, 2, 0))
    ax.imshow(map_clr(y_predv[vl_idx[i]][1].argmax(axis=0).numpy()), alpha=0.5)
    plt.savefig(f"{dst_back}/{img_back}")
    print(f"Saved to {dst_back}/{img_back}\n")

    # Delete the figure after saving to save memory
    plt.close(fig)

def main(threading):
    print("=================================================")
    print("INFERENCING STARTED")

    # Load the model
    kupu = BtrflyNet()
    kupu.load_state_dict(
        torch.load("./model-eff0406a.pt", map_location=torch.device("cpu"))
    )
    print("Model loaded\n")

    # Get the files
    src_front = "./imgs/wholeBodyANT_preprocessed"
    src_back = "./imgs/wholeBodyPOST_preprocessed"
    files_front = natsorted(os.listdir(src_front))
    files_back = natsorted(os.listdir(src_back))

    # If src contains "preprocessed", replace it with "processed", else, append "_processed"
    dst_front = src_front.replace("preprocessed", "processed") if "preprocessed" in src_front else f"{src_front}_processed"
    dst_back = src_back.replace("preprocessed", "processed") if "preprocessed" in src_back else f"{src_back}_processed"

    # Check if dst exists
    if not exists(dst_front):
        os.makedirs(dst_front)
        print(f"Created {dst_front}")
    if not exists(dst_back):
        os.makedirs(dst_back)
        print(f"Created {dst_back}")

    if threading:
        max_threads = 8  # Set the number of threads you want to use
        print(f"Program running in threaded mode with {max_threads} threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Start inference
            futures = []
            for i, (ff, fb) in enumerate(zip(files_front, files_back)):
                future = executor.submit(render_data, src_front, src_back, dst_front, dst_back, ff, fb, kupu, i)
                futures.append(future)

            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                # Handle any exceptions that might have occurred during rendering
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}\n")
    else:
        print("Program running in non-threaded mode")
        for ff, fb in zip(files_front, files_back):
            render_data(src_front, src_back, dst_front, dst_back, ff, fb, kupu)

    # Get the number of files
    num_files = len(files_front)
    num_files_processed = len(os.listdir(dst_front))

    # Write report
    print("=================================================")
    print("INFERENCING DONE")
    print(f"Number of images: {num_files}")
    print(f"Number of images processed: {num_files_processed}")
    with open("report.txt", "w") as f:
        f.write(f"Number of images: {num_files}\n")
        f.write(f"Number of images processed: {num_files_processed}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the inference script.")
    parser.add_argument("--threading", action="store_true", help="Run the program in threaded mode.")
    args = parser.parse_args()

    main(args.threading)
