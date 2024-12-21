# target.py
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No image path provided.")
        sys.exit(1) 

    image_path = sys.argv[1]
    print(f"the picture is {image_path}")
