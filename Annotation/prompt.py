prompt="""
You are a language model. I will show you a JSON file with 20 items. 
For each item, there are four keys: "id," "product," "prediction," and "ground truth." 
You need to evaluate each item and provide me with a new JSON file. 
For each item, add two extra keys: "score_vis" and "score_guidance."

The "score_vis" should be either 0 or 1. 
If, according to the prediction, it can "see" the product, then "score_vis" should be set to 1; 
otherwise, it should be set to 0.

For "score_guidance," the value should range between 0 and 1. 
To determine this value, we need to compare the prediction with the ground truth. 
The more the prediction provides guidance that close aligns with the user's desired product and the ground truth, the score is higher. 


Concerning "score_guidance," its value should fall within the range of 0 to 1. 
To determine this value, compare the prediction with the ground truth. 
The higher the prediction aligns with the user's desired product and the ground truth, the higher the score. 
It's worth noting that predictions may sometimes be overly general or even incorrect. 
In such cases, where the prediction doesn't strongly support the user's desired product (i.e., the user might not find the desired product based on the prediction), 
the "score_guidance" should not exceed 0.5.
Conversely, if the prediction is accurate and effectively guides the user to their desired product, set "score_guidance" closer to 1.
You can choose a value from the following range: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] to represent the "score_guidance" accordingly.

Please return the JSON format as follows: [{"id", "product", "prediction", "ground truth", "score_vis", "score_guidance"}, {...}].
"""