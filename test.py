import os
import random
import json
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_objects_with_labels(folder_path):
    """
    Extract objects and their corresponding labels from input JSON files and images.

    Parameters:
        folder_path (str): Path to the folder containing images and JSON files.

    Returns:
        list: A list of tuples (object_image, object_mask).
        list: A list of labels corresponding to each object.
    """
    objects = []
    labels = []
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file), "r") as f:
            data = json.load(f)

        # Find the corresponding image
        image_file_name = os.path.splitext(json_file)[0] + ".jpeg"
        image_path = os.path.join(folder_path, image_file_name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image {image_file_name} not found. Skipping.")
            continue

        # Process each shape in the JSON file
        for shape in data.get("shapes", []):
            points = np.array(shape["points"], dtype=np.int32)
            mask = np.zeros((data["imageHeight"], data["imageWidth"]), dtype=np.uint8)

            # Create a binary mask for the current object
            cv2.fillPoly(mask, [points], 255)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            objects.append((masked_image, mask))
            labels.append(shape["label"])

    return objects, labels


def place_objects_on_canvas(objects, output_size, num_objects_range, min_visible, scale_range, labels):
    """
    Randomly place objects onto a blank canvas.

    Parameters:
        objects (list): A list of tuples (object_image, object_mask).
        output_size (tuple): Size of the canvas (width, height).
        num_objects_range (tuple): Minimum and maximum number of objects to place.
        min_visible (float): Minimum percentage of the object that must remain visible.
        scale_range (tuple): Range of scaling factors for resizing objects.
        labels (list): List of labels corresponding to the objects.

    Returns:
        ndarray: The canvas with objects placed.
        ndarray: The combined mask for all objects.
        list: Metadata about each placed object (label, position, contour).
    """
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    combined_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    object_metadata = []

    # Determine the number of objects to place
    min_objects, max_objects = num_objects_range
    num_objects_to_place = random.randint(min_objects, min(max_objects, len(objects)))
    selected_indices = random.sample(range(len(objects)), num_objects_to_place)

    for obj_idx in selected_indices:
        obj_image, obj_mask = objects[obj_idx]
        h, w = obj_image.shape[:2]

        # Randomly scale the object
        scale_factor = np.random.uniform(*scale_range)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        obj_image = cv2.resize(obj_image, (new_w, new_h))
        obj_mask = cv2.resize(obj_mask, (new_w, new_h))

        # Randomly place the object on the canvas
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

            # Compute contours for metadata
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


def generate_background_with_inpainting(control_mask, prompt, output_size):
    """
    Generate a realistic background using Stable Diffusion Inpainting.

    Parameters:
        control_mask (ndarray): Combined binary mask of all objects.
        prompt (str): Text prompt describing the desired background.
        output_size (tuple): Size of the output image (width, height).

    Returns:
        ndarray: The generated inpainted image, resized to match `output_size`.
    """
    inpainting_model = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting_model)
    pipe = pipe.to(DEVICE)

    # Create a blank white canvas as the base
    blank_canvas = np.ones((control_mask.shape[0], control_mask.shape[1], 3), dtype=np.uint8) * 255
    blank_canvas_pil = Image.fromarray(blank_canvas)

    control_image = Image.fromarray((control_mask > 0).astype(np.uint8) * 255).convert("L")

    try:
        result = pipe(
            prompt=prompt,
            image=blank_canvas_pil,
            mask_image=control_image,
            guidance_scale=7.5
        ).images[0]
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return None

    # Resize the result to match `output_size`
    try:
        result = result.resize(output_size, resample=Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error resizing the generated image: {e}")
        return None

    return np.array(result)


def save_metadata_as_json(output_path, output_size, object_metadata):
    """
    Save metadata about placed objects as a JSON file.

    Parameters:
        output_path (str): Path to save the JSON file.
        output_size (tuple): Size of the output image (width, height).
        object_metadata (list): Metadata for each placed object.
    """
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
    """
    Blend objects into the background with softened edges.

    Parameters:
        background (ndarray): Generated background image.
        canvas (ndarray): Canvas with placed objects.
        combined_mask (ndarray): Combined binary mask of all objects.
        blur_size (int): Size of Gaussian blur for edge softening.

    Returns:
        ndarray: The final blended image.
    """
    # Debug: Print dimensions for validation
    print(f"Background shape: {background.shape}")
    print(f"Canvas shape: {canvas.shape}")
    print(f"Mask shape: {combined_mask.shape}")

    # Ensure mask matches the background dimensions
    if combined_mask.shape[:2] != background.shape[:2]:
        combined_mask = cv2.resize(combined_mask, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create softened edges
    mask = (combined_mask > 0).astype(np.uint8) * 255
    softened_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    softened_mask = softened_mask / 255.0

    softened_mask = np.expand_dims(softened_mask, axis=-1)  # Expand to match RGB
    blended_image = background * (1 - softened_mask) + canvas * softened_mask
    return blended_image.astype(np.uint8)


def main():
    config = {
        "INPUT_FOLDER": "./images",
        "OUTPUT_FOLDER": "./output",
        "OUTPUT_SIZE": (400, 300),
        "SCALE_RANGE": (0.05, 0.15),
        "MIN_VISIBLE": 0.5,
        "NUM_OBJECTS_RANGE": (1, 5),
        "PROMPTS": ["a realistic kitchen table.", "a wooden desk.", "a marble countertop."],
        "NUM_IMAGES": 10
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
        background = generate_background_with_inpainting(combined_mask, prompt, config["OUTPUT_SIZE"])
        if background is None:
            print(f"Skipping generation {i} due to background generation failure.")
            continue

        output_image_path = os.path.join(config["OUTPUT_FOLDER"], f"generated_{i}.jpg")
        final_image = overlay_objects_with_soft_edges(background, canvas, combined_mask)
        cv2.imwrite(output_image_path, final_image)
        save_metadata_as_json(output_image_path, config["OUTPUT_SIZE"], object_metadata)


if __name__ == "__main__":
    main()
