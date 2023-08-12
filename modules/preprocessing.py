from PIL import Image, ImageOps, ImageEnhance
import cv2
import os
import shutil
import argparse
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

size = (128, 512) # based on model engine
factor = 3 # based on user (trial and error)
threshold = 0.55 # based on user (trial and error)
baseline = cv2.imread("./imgs/baseReferenceANT/indo-011.png")

error_imgs_count = 0
error_imgs = []
dst = ""

def adjust(img, src, i):
    try:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Perform contour detection
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest contour (assuming it corresponds to the skeleton/object)
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Define the cropping box (left, upper, right, lower)
        cropping_box = (x, y, x + w, y + h)
        # Crop the image
        pil_image = Image.open(f"{src}/{i}.jpg")
        cropped_image = pil_image.crop(cropping_box)
        # Resize the image
        resized = cropped_image.resize(size)
        # Invert the image
        inverted = ImageOps.invert(resized)
        # Enhance the image
        enhancer = ImageEnhance.Contrast(inverted)
        output = enhancer.enhance(factor)

        bit = output.convert("RGB", palette=Image.Palette.ADAPTIVE)

        # convert bit into cv2 format for analysis
        cv2_bit = cv2.cvtColor(np.array(bit), cv2.COLOR_RGB2BGR)
        analyzed = analyze(cv2_bit, baseline, threshold)

        if analyzed == 'Accepted':
            bit.save(f"{dst}_accepted/{i}.png")
            printed_dst = f"{dst}_accepted"
        else:
            bit.save(f"{dst}_rejected/{i}.png")
            printed_dst = f"{dst}_rejected"
        
        printed_src = src.split("/")[-1]
        print(f"Image {i} from {printed_src} saved to {printed_dst}/{i}.png")

    except Exception as e:
        print(f"Image error in {i}\n{e}")
        global error_imgs_count
        error_imgs_count += 1
        error_img_name = f"{src}/{i}.jpg"
        error_imgs.append(error_img_name)
        # Write the error to a file
        with open(f"{src}_preprocessed_errors/log/log.txt", "a") as f:
            f.write(f"Image error in {i}.jpg\n{e}\n")


def analyze(img, baseline, threshold):
    # Convert images to grayscale
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    # Calculate structural similarity index
    ssim_index = compare_ssim(grayscale_image, grayscale_baseline)
    # Determine whether the image is a full skeleton based on SSIM index
    if ssim_index >= threshold:
        return 'Accepted'
    else:
        return 'Rejected'
    

def process_images(src):
    # Save all the images names in a list
    imgs = []
    for name in os.listdir(f"{src}"):
        if name.endswith(".jpg"):
            imgs.append(name)
    for im in imgs:
        img = cv2.imread(f"{src}/{im}")
        adjust(img, src, im[:-4])

def main():
    global dst 
    global error_imgs_count 
    global error_imgs 
    global size 
    global factor
    global threshold
    global baseline

    parser = argparse.ArgumentParser(description="Image preprocessing script")
    parser.add_argument("--src_front", required=True, help="Path to the source folder for front images")
    parser.add_argument("--src_back", required=True, help="Path to the source folder for back images")
    args = parser.parse_args()

    print("=================================================")
    print("PREPROCESSING STARTED\n")

    src_front = args.src_front
    src_back = args.src_back
    srcs = [src_front, src_back]

    for i in range(len(srcs)):
        # Create the error folder if it doesn't exist
        if not os.path.exists(f"{srcs[i]}_preprocessed_errors"):
            os.makedirs(f"{srcs[i]}_preprocessed_errors")
        else:
            # Delete previous error images
            shutil.rmtree(f"{srcs[i]}_preprocessed_errors")
            os.makedirs(f"{srcs[i]}_preprocessed_errors")
        if not os.path.exists(f"{srcs[i]}_preprocessed_errors/log"):
            os.makedirs(f"{srcs[i]}_preprocessed_errors/log")
        # Check if the log file exists
        if os.path.exists(f"{srcs[i]}_preprocessed_errors/log/log.txt"):
            # Delete previous log file
            with open(f"{srcs[i]}_preprocessed_errors/log/log.txt", "w") as f:
                f.write("")

        dst = f"{srcs[i]}_preprocessed"
        # Create the folder if it doesn't exist
        if not os.path.exists(f"{dst}_accepted") or os.path.exists(f"{dst}_rejected"):
            os.makedirs(f"{dst}_accepted")
            os.makedirs(f"{dst}_rejected")
        else:
            # Delete previous preprocessed images
            shutil.rmtree(f"{dst}_accepted")
            shutil.rmtree(f"{dst}_rejected")
            os.makedirs(f"{dst}_accepted")
            os.makedirs(f"{dst}_rejected")

        if i == 0:
            print("Preprocessing front images...")
        else:
            print("Preprocessing back images...")
        process_images(srcs[i])

        # Error count
        print(f"Number of images with errors: {error_imgs_count}\n")
        if error_imgs_count > 0:
            print(f"Error images: {error_imgs}")
            # Write the error count to the first line of the error file
            with open(f"{srcs[i]}_preprocessed_errors/log/log.txt", "r+") as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f"Number of images with errors: {error_imgs_count}\n\n" + content)
            # Copy the error images to a separate folder
            print(f"Copying error images to {srcs[i]}_preprocessed_errors...")
            if not os.path.exists(f"{srcs[i]}_preprocessed_errors"):
                os.makedirs(f"{srcs[i]}_preprocessed_errors")
            for error_img in error_imgs:
                try:
                    shutil.copy(error_img, f"{srcs[i]}_preprocessed_errors")
                except Exception as e:
                    print(f"Error copying {error_img}\n{e}")
        # If there is no error in the log file, delete the log folder
        if len(os.listdir(f"{srcs[i]}_preprocessed_errors/log")) == 0:
            shutil.rmtree(f"{srcs[i]}_preprocessed_errors/log")
        # If there is no error, delete the folder
        if len(os.listdir(f"{srcs[i]}_preprocessed_errors")) == 0:
            shutil.rmtree(f"{srcs[i]}_preprocessed_errors")

        # Reset the error count and list
        error_imgs_count = 0
        error_imgs = []

    print(f"Number of accepted images processed: {len(os.listdir(f'{src_front}_preprocessed_accepted')) + len(os.listdir(f'{src_back}_preprocessed_accepted'))}")
    print(f"Number of rejected images processed: {len(os.listdir(f'{src_front}_preprocessed_rejected')) + len(os.listdir(f'{src_back}_preprocessed_rejected'))}")
    print("PREPROCESSING DONE")
    print("=================================================")

if __name__ == "__main__":
    main()
