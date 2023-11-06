import json
import random

# Load the JSON data from a file
with open('transformed.json', 'r') as file:
    data = json.load(file)

# Shuffle the data to ensure randomness
random.shuffle(data)

# The total number of items you want in the test set
num_test_items = 20

# Split the data into training and test sets
test_data = data[:num_test_items]
train_data = data[num_test_items:num_test_items+65]

# Verify the length of the items, if needed
print(f"Total items: {len(data)}")
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# Save the training data to ft.json
with open('ft.json', 'w') as file:
    json.dump(train_data, file, indent=4)

# Save the test data to ts.json
with open('ts.json', 'w') as file:
    json.dump(test_data, file, indent=4)

print("Data split into training and test files.")