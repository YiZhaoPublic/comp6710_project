import json

# Read the original JSON data
with open('vqa_en.json', 'r') as file:
    original_data = json.load(file)


# Function to transform data to the target format
def transform_data(original_data):
    transformed_data = []
    sub_rm="For instance, I could take about 5 steps forward, then turn 90 degrees to the right, take about 3 steps forward, and then reach up to the right to get the milk."
    for item in original_data:
        # Construct the image path using the Image_index
        image_path = f"/mnt/zy/projects/report/resized_images_1024_768/{item['Image_index']}.jpg"

        # Create the structure for the transformed data
        transformed_item = {
            "id": f"identity_{item['Image_index']}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"Picture 1: <img>{image_path}</img>\n{item['Question'].replace(sub_rm, '')}"
                },
                {
                    "from": "assistant",
                    "value": item['Answer']
                }
            ]
        }
        transformed_data.append(transformed_item)
    return transformed_data

#"Picture 1: <img>resized_images_1024_768/0.jpg</img>\n
# Hello, I am visually impaired and need your assistance to complete the task of 'obtaining a product‘.
# Here is where I stand, and the scene depicted in the image is the view in front of me.
# Please provide a guide for me to obtain the product I need based on the scene in the image.
# First, please tell me: is the product I need present in the field of view of the image?
# Second, please tell me: is there a walkable passage in front of me?
# Third, please tell me: how should I go about obtaining the product I need?
# The product I need is: soy souce, please provide guidance for me based on the scene in the image."

# Perform the transformation
transformed_data = transform_data(original_data)

# Write the new JSON data to a file
with open('/mnt/zy/projects/report/Annotation/transformed.json', 'w') as file:
    json.dump(transformed_data, file, indent=2)

print("Transformation complete. Data saved to 'transformed.json'.")