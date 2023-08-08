import cv2
from skimage.metrics import structural_similarity as compare_ssim
import os
from natsort import natsorted

def is_full_skeleton(image, baseline_image, threshold):
    # Convert images to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_baseline = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

    # Calculate structural similarity index
    ssim_index = compare_ssim(grayscale_image, grayscale_baseline)

    # Determine whether the image is a full skeleton based on SSIM index
    if ssim_index >= threshold:
        return f"Full Skeleton ({ssim_index})"
    else:
        return f"Cropped or No Skeleton ({ssim_index})"
    
def gather_compared_image_paths(folder_path):
    compared_image_paths = []
    for filename in natsorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            compared_image_paths.append(image_path)
    return compared_image_paths

def main():
    # Load baseline full skeleton image
    baseline_image = cv2.imread("../kupu_offline/model/img/indo-011.png")

    # Load and process compared images
    compared_folder = "../img/editedWholeBodyANT"
    threshold = 0.55

    compared_image_paths = gather_compared_image_paths(compared_folder)
    
    # Create a dictionary to store results
    results_dict = {}

    for image_path in compared_image_paths:
        compared_image = cv2.imread(image_path)
        result = is_full_skeleton(compared_image, baseline_image, threshold)
        print(f"Image: {image_path}, Result: {result}")

        # Extract the filename from the path
        filename = os.path.basename(image_path)
        
        # Store result in the dictionary
        results_dict[filename] = result

    return results_dict

if __name__ == "__main__":
    results = main()
