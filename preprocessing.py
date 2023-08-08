from PIL import Image, ImageOps, ImageEnhance
import cv2
import os
import shutil

size = (128, 512)
factor = 3

error_imgs_count = 0
error_imgs = []

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
        bit.save(f"{dst}/{i}.png")
        print(f"Image {i} from {src} saved to {dst}/{i}.png")

    except Exception as e:
        print(f"Image error in {i}\n{e}")
        global error_imgs_count
        error_imgs_count += 1
        error_img_name = f"{src}/{i}.jpg"
        error_imgs.append(error_img_name)
        # Write the error to a file
        with open(f"{src}_preprocessed_errors/log/log.txt", "a") as f:
            f.write(f"Image error in {i}.jpg\n{e}\n")

def process_images(src):
    # Save all the images names in a list
    imgs = []
    for name in os.listdir(f"{src}"):
        if name.endswith(".jpg"):
            imgs.append(name)
            print(f"Image {name} added to the list")
    print(f"Images' list: {imgs}")

    for im in imgs:
        img = cv2.imread(f"{src}/{im}")
        adjust(img, src, im[:-4])

print("=================================================")
print("PREPROCESSING STARTED")

src_front = "imgs/wholeBodyANT"
src_back = "imgs/wholeBodyPOST"
srcs = [src_front, src_back]

for src in srcs:
    # Create the error folder if it doesn't exist
    if not os.path.exists(f"{src}_preprocessed_errors"):
        os.makedirs(f"{src}_preprocessed_errors")
    else:
        # Delete previous error images
        shutil.rmtree(f"{src}_preprocessed_errors")
        os.makedirs(f"{src}_preprocessed_errors")
    if not os.path.exists(f"{src}_preprocessed_errors/log"):
        os.makedirs(f"{src}_preprocessed_errors/log")
    # Check if the log file exists
    if os.path.exists(f"{src}_preprocessed_errors/log/log.txt"):
        # Delete previous log file
        with open(f"{src}_preprocessed_errors/log/log.txt", "w") as f:
            f.write("")

    dst = f"{src}_preprocessed"
    # Create the folder if it doesn't exist
    if not os.path.exists(f"{dst}"):
        os.makedirs(f"{dst}")
    else:
        # Delete previous preprocessed images
        shutil.rmtree(f"{dst}")
        os.makedirs(f"{dst}")
    process_images(src)

    # Error count
    print(f"Number of images with errors: {error_imgs_count}")
    if error_imgs_count > 0:
        print(f"Error images: {error_imgs}")
        # Write the error count to the first line of the error file
        with open(f"{src}_preprocessed_errors/log/log.txt", "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"Number of images with errors: {error_imgs_count}\n\n" + content)
        # Copy the error images to a separate folder
        print(f"Copying error images to {src}_preprocessed_errors...")
        if not os.path.exists(f"{src}_preprocessed_errors"):
            os.makedirs(f"{src}_preprocessed_errors")
        for error_img in error_imgs:
            try:
                shutil.copy(error_img, f"{src}_preprocessed_errors")
            except Exception as e:
                print(f"Error copying {error_img}\n{e}")
    # If there is no error in the log file, delete the log folder
    if len(os.listdir(f"{src}_preprocessed_errors/log")) == 0:
        shutil.rmtree(f"{src}_preprocessed_errors/log")
    # If there is no error, delete the folder
    if len(os.listdir(f"{src}_preprocessed_errors")) == 0:
        shutil.rmtree(f"{src}_preprocessed_errors")

    # Reset the error count and list
    error_imgs_count = 0
    error_imgs = []

print(f"\nNumber of images processed: {len(os.listdir(src_front))}")
print("PREPROCESSING DONE")
print("=================================================")
