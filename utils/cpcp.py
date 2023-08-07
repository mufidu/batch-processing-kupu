import sys
import shutil

if len(sys.argv) != 3:
    print("Usage: python app.py <command> <filename>")
    sys.exit(1)

command = sys.argv[1]
filename = sys.argv[2]

if command == 'cp':
    src_paths = [
        f"../imgs/wholeBodyANT/{filename}.jpg",
        f"../imgs/wholeBodyPOST/{filename}.jpg"
    ]
    dest_paths = [
        f"imgs/wholeBodyANT/{filename}.jpg",
        f"imgs/wholeBodyPOST/{filename}.jpg"
    ]

    for src, dest in zip(src_paths, dest_paths):
        shutil.copy(src, dest)
    print("Files copied successfully.")
else:
    print("Invalid command. Use 'cp' command.")
    sys.exit(1)
