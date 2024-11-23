import os
import json
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import base64

# Configuration
OUTPUT_SIZE = (480, 480)  # Output image size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Read JSON files and extract objects with their masks
def extract_objects_from_folder(folder_path):
    objects = []
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file), "r") as f:
            data = json.load(f)

        # Read the corresponding image path
        if "imagePath" not in data:
            print(f"Key 'imagePath' missing in {json_file}. Skipping.")
            continue

        image_path = os.path.join(folder_path, data["imagePath"])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image {data['imagePath']} not found. Skipping.")
            continue

        # Iterate through all shapes in the JSON
        for shape in data.get("shapes", []):
            if "points" not in shape:
                print(f"Key 'points' missing in a shape in {json_file}. Skipping shape.")
                continue

            points = np.array(shape["points"], dtype=np.int32)
            mask = np.zeros((data["imageHeight"], data["imageWidth"]), dtype=np.uint8)

            # Create a mask for the current shape
            cv2.fillPoly(mask, [points], 255)

            # Extract the object using the mask
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            objects.append((masked_image, mask))

    return objects


# Step 2: Place objects randomly on the canvas
def place_objects_on_canvas(objects, output_size, min_visible=0.5, scale_range=(0.5, 1.0)):
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    combined_mask = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    object_metadata = []

    for obj_idx, (obj_image, obj_mask) in enumerate(objects):
        h, w = obj_image.shape[:2]

        # Randomly scale the object
        scale_factor = np.random.uniform(*scale_range)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        obj_image = cv2.resize(obj_image, (new_w, new_h))
        obj_mask = cv2.resize(obj_mask, (new_w, new_h))

        # Ensure object is at least min_visible visible on the canvas
        max_offset_x = output_size[0] - int(new_w * min_visible)
        max_offset_y = output_size[1] - int(new_h * min_visible)
        min_offset_x = -int(new_w * (1 - min_visible))
        min_offset_y = -int(new_h * (1 - min_visible))

        x_offset = np.random.randint(min_offset_x, max_offset_x + 1)
        y_offset = np.random.randint(min_offset_y, max_offset_y + 1)

        # Calculate the region to place the object
        x_start, x_end = max(x_offset, 0), min(x_offset + new_w, output_size[0])
        y_start, y_end = max(y_offset, 0), min(y_offset + new_h, output_size[1])

        # Calculate the corresponding region on the object
        obj_x_start = max(0, -x_offset)
        obj_x_end = obj_x_start + (x_end - x_start)
        obj_y_start = max(0, -y_offset)
        obj_y_end = obj_y_start + (y_end - y_start)

        # Place the visible part of the object on the canvas
        if x_start < x_end and y_start < y_end:
            region = canvas[y_start:y_end, x_start:x_end]
            mask_region = combined_mask[y_start:y_end, x_start:x_end]

            object_subregion = obj_image[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
            mask_subregion = obj_mask[obj_y_start:obj_y_end, obj_x_start:obj_x_end]

            region[mask_subregion > 0] = object_subregion[mask_subregion > 0]
            mask_region[mask_subregion > 0] = 255

            # Record metadata about the placed object
            object_points = [
                [x_start, y_start],
                [x_end, y_start],
                [x_end, y_end],
                [x_start, y_end]
            ]
            object_metadata.append({
                "label": f"object_{obj_idx}",
                "points": object_points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    return canvas, combined_mask, object_metadata


# Step 3: Save metadata as JSON
def save_metadata_as_json(output_path, output_size, object_metadata):
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_data = encode_image_to_base64(output_path)

    json_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": object_metadata,
        "imagePath": os.path.basename(output_path),
        "imageData": image_data,
        "imageHeight": output_size[1],
        "imageWidth": output_size[0]
    }

    json_file_path = output_path.replace(".jpg", ".json")
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON metadata saved at: {json_file_path}")


# Step 4: Generate background using ControlNet
def generate_background_with_controlnet_only(control_mask, prompt):
    base_model = "runwayml/stable-diffusion-v1-5"
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model, controlnet=controlnet)
    pipe = pipe.to(DEVICE)

    control_image = Image.fromarray(control_mask).convert("L")
    result = pipe(prompt, image=control_image, guidance_scale=7.5).images[0]
    return np.array(result)


# Step 5: Blend objects with soft edges
def overlay_objects_with_soft_edges(background, canvas, combined_mask, blur_size=5):
    mask = (combined_mask > 0).astype(np.uint8) * 255
    softened_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0) / 255.0
    softened_mask = np.expand_dims(softened_mask, axis=-1)

    blended_image = background * (1 - softened_mask) + canvas * softened_mask
    return blended_image.astype(np.uint8)


# Main function
def main():
    folder_path = "./images"
    folder_out_path = "./output"
    objects = extract_objects_from_folder(folder_path)

    if not objects:
        print("No valid objects found in the folder.")
        return

    canvas, combined_mask, object_metadata = place_objects_on_canvas(
        objects, OUTPUT_SIZE, min_visible=0.5, scale_range=(0.1, 0.25)
    )

    canvas_path = os.path.join(folder_out_path, "canvas_preview.jpg")
    cv2.imwrite(canvas_path, canvas)

    prompt = "a realistic kitchen table."
    background = generate_background_with_controlnet_only(combined_mask, prompt)

    final_image = overlay_objects_with_soft_edges(background, canvas, combined_mask)
    output_path = os.path.join(folder_out_path, "output_combined.jpg")
    cv2.imwrite(output_path, final_image)

    save_metadata_as_json(output_path, OUTPUT_SIZE, object_metadata)


if __name__ == "__main__":
    main()
