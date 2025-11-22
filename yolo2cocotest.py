import os
import json
from PIL import Image

def yolo_to_custom_format(yolo_dir, output_file, image_dir):
    """
    Convert YOLO format annotations to the desired JSON format without score.

    :param yolo_dir: Directory containing YOLO annotation files.
    :param output_file: Output JSON file.
    :param image_dir: Directory containing corresponding images.
    """
    results = []

    for yolo_file in sorted(os.listdir(yolo_dir)):
        if not yolo_file.endswith(".txt"):
            continue

        # Get image ID
        image_id = os.path.splitext(yolo_file)[0]

        # Get image dimensions
        image_path = os.path.join(image_dir, image_id + ".jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist.")
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Parse YOLO annotation file
        with open(os.path.join(yolo_dir, yolo_file), "r") as f:
            for line in f:
                try:
                    category_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
                    category_id = int(category_id)

                    # Convert YOLO normalized bbox to absolute values
                    x_center *= width
                    y_center *= height
                    bbox_width *= width
                    bbox_height *= height

                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2

                    # Append the result
                    results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [round(x_min, 3), round(y_min, 3), round(bbox_width, 3), round(bbox_height, 3)]
                    })
                except Exception as e:
                    print(f"Error parsing annotation in {yolo_file}: {e}")

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

# Example usage
yolo_dir = "/home/qk/data/new_test/labels"
image_dir = "/home/qk/data/new_test/images"
output_file = "/home/qk/data/new_test/annotations.json"

yolo_to_custom_format(yolo_dir, output_file, image_dir)
