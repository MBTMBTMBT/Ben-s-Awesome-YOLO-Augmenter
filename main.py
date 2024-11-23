import os
import random
import json
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_objects_with_labels(folder_path):
    objects = []
    labels = []
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file), "r") as f:
            data = json.load(f)

        image_file_name = os.path.splitext(json_file)[0] + ".jpeg"
        image_path = os.path.join(folder_path, image_file_name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image {image_file_name} not found. Skipping.")
            continue

        for shape in data.get("shapes", []):
            points = np.array(shape["points"], dtype=np.int32)
            mask = np.zeros((data["imageHeight"], data["imageWidth"]), dtype=np.uint8)

            cv2.fillPoly(mask, [points], 255)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            objects.append((masked_image, mask))
            labels.append(shape["label"])

    return objects, labels


def place_objects_on_canvas(objects, output_size, num_objects_range, min_visible, scale_range, labels):
    """
    Randomly place a variable number of objects on the canvas.

    Parameters:
        objects (list): List of tuples (image, mask) for each object.
        output_size (tuple): Size of the canvas (width, height).
        num_objects_range (tuple): A tuple (min_objects, max_objects) defining the range of object counts.
        min_visible (float): Minimum percentage of the object that must be visible on the canvas.
        scale_range (tuple): Range of scale factors for resizing objects.
        labels (list): List of labels corresponding to the objects.

    Returns:
        canvas (ndarray): Canvas with placed objects.
        combined_mask (ndarray): Combined mask of all objects.
        object_metadata (list): Metadata about placed objects, including labels and contours.
    """
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    combined_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    object_metadata = []

    # Determine the random number of objects to place
    min_objects, max_objects = num_objects_range
    num_objects_to_place = random.randint(min_objects, min(max_objects, len(objects)))

    # Randomly select indices for the chosen number of objects
    selected_indices = random.sample(range(len(objects)), num_objects_to_place)

    for obj_idx in selected_indices:
        obj_image, obj_mask = objects[obj_idx]
        h, w = obj_image.shape[:2]

        scale_factor = np.random.uniform(*scale_range)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        obj_image = cv2.resize(obj_image, (new_w, new_h))
        obj_mask = cv2.resize(obj_mask, (new_w, new_h))

        max_offset_x = output_size[0] - int(new_w * min_visible)
        max_offset_y = output_size[1] - int(new_h * min_visible)
        min_offset_x = -int(new_w * (1 - min_visible))
        min_offset_y = -int(new_h * (1 - min_visible))

        x_offset = np.random.randint(min_offset_x, max_offset_x + 1)
        y_offset = np.random.randint(min_offset_y, max_offset_y + 1)

        x_start, x_end = max(x_offset, 0), min(x_offset + new_w, output_size[0])
        y_start, y_end = max(y_offset, 0), min(y_offset + new_h, output_size[1])

        obj_x_start = max(0, -x_offset)
        obj_x_end = obj_x_start + (x_end - x_start)
        obj_y_start = max(0, -y_offset)
        obj_y_end = obj_y_start + (y_end - y_start)

        if x_start < x_end and y_start < y_end:
            region = canvas[y_start:y_end, x_start:x_end]
            mask_region = combined_mask[y_start:y_end, x_start:x_end]

            object_subregion = obj_image[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
            mask_subregion = obj_mask[obj_y_start:obj_y_end, obj_x_start:obj_x_end]

            region[mask_subregion > 0] = object_subregion[mask_subregion > 0]
            mask_region[mask_subregion > 0] = 255

            single_mask = np.zeros_like(mask_region, dtype=np.uint8)
            single_mask[mask_subregion > 0] = 255
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.reshape(-1, 2).tolist() for c in contours]

            for contour in contours:
                for point in contour:
                    point[0] += x_start
                    point[1] += y_start

            object_metadata.append({
                "label": labels[obj_idx],
                "points": contours[0] if contours else []
            })

    return canvas, combined_mask, object_metadata


def generate_background_with_controlnet_only(control_mask, prompt, output_size):
    base_model = "runwayml/stable-diffusion-v1-5"
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model, controlnet=controlnet)
    pipe = pipe.to(DEVICE)

    control_image = Image.fromarray(control_mask).convert("L")
    result = pipe(prompt, image=control_image, guidance_scale=7.5).images[0]
    result = result.resize(output_size, resample=Image.Resampling.LANCZOS)
    return np.array(result)


def save_metadata_as_json(output_path, output_size, object_metadata):
    shapes = []
    for obj in object_metadata:
        shapes.append({
            "label": obj["label"],
            "text": "",
            "points": [[float(x), float(y)] for x, y in obj["points"]],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    json_data = {
        "version": "0.4.15",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(output_path),
        "imageData": None,
        "imageHeight": output_size[1],
        "imageWidth": output_size[0]
    }

    json_file_path = output_path.replace(".jpg", ".json")
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=4)


def overlay_objects_with_soft_edges(background, canvas, combined_mask, blur_size=5):
    mask = (combined_mask > 0).astype(np.uint8) * 255
    softened_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    softened_mask = softened_mask / 255.0

    softened_mask = np.expand_dims(softened_mask, axis=-1)
    blended_image = background * (1 - softened_mask) + canvas * softened_mask
    return blended_image.astype(np.uint8)


def main():
    # Configurations
    config = {
        "INPUT_FOLDER": "./images",  # Input folder containing images and JSONs
        "OUTPUT_FOLDER": "./output",  # Output folder for generated images
        "OUTPUT_SIZE": (400, 300),  # Size of the output canvas (width, height)
        "SCALE_RANGE": (0.05, 0.15),  # Range for scaling small objects
        "MIN_VISIBLE": 0.5,  # Minimum percentage of the object visible on the canvas
        "NUM_OBJECTS_RANGE": (1, 5),  # Random range for the number of objects to place
        "PROMPTS": ["a realistic kitchen table.", "a wooden desk.", "a marble countertop."],
        "NUM_IMAGES": 10,  # Total number of images to generate
    }

    if not os.path.exists(config["OUTPUT_FOLDER"]):
        os.makedirs(config["OUTPUT_FOLDER"])

    objects, labels = extract_objects_with_labels(config["INPUT_FOLDER"])
    if not objects:
        print("No valid objects found.")
        return

    for i in range(config["NUM_IMAGES"]):
        canvas, combined_mask, object_metadata = place_objects_on_canvas(
            objects,
            config["OUTPUT_SIZE"],
            config["NUM_OBJECTS_RANGE"],
            config["MIN_VISIBLE"],
            config["SCALE_RANGE"],
            labels
        )

        prompt = random.choice(config["PROMPTS"])
        background = generate_background_with_controlnet_only(combined_mask, prompt, config["OUTPUT_SIZE"])

        output_image_path = os.path.join(config["OUTPUT_FOLDER"], f"generated_{i}.jpg")
        final_image = overlay_objects_with_soft_edges(background, canvas, combined_mask)
        cv2.imwrite(output_image_path, final_image)

        save_metadata_as_json(output_image_path, config["OUTPUT_SIZE"], object_metadata)


if __name__ == "__main__":
    main()
