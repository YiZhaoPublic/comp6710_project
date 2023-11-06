from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import AutoPeftModelForCausalLM
import json
import os

def inference():
    torch.manual_seed(1234)
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat-Int4", device_map="cuda", trust_remote_code=True).eval()
    query = tokenizer.from_list_format([
        {'image': './resized_images_1024_768/5.jpg'},
        {'text': '这是什么?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)
    response, history = model.chat(tokenizer, '框出图中食物的位置', history=history)
    print(response)
    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
      image.save('1.jpg')
    else:
      print("no box")
    return None

def quantization_inference():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat-Int4",device_map="cuda",trust_remote_code=True).eval()
    # Either a local path or an url between <img></img> tags.
    image_path = './resized_images_1024_768/5.jpg'
    response, history = model.chat(tokenizer, query=f'<img>{image_path}</img>这是什么', history=None)
    print(response)
    return None


def after_ft_inference():
    path_to_adapter="Qwen-VL/output_qwen"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(path_to_adapter, device_map="cuda",trust_remote_code=True).eval()
    # image_path = './resized_images_1024_768/5.jpg'
    # response, history = model.chat(tokenizer, query=f'<img>{image_path}</img>这是什么', history=None)


    query="Picture 1: <img>/mnt/zy/projects/report/resized_images_1024_768/37.jpg</img>\nHello, I am visually impaired and need your assistance to complete the task of 'obtaining a product\u2018.Here is where I stand, and the scene depicted in the image is the view in front of me.Please provide a guide for me to obtain the product I need based on the scene in the image.First, please tell me: is the product I need present in the field of view of the image?Second, please tell me: is there a walkable passage in front of me?Third, please tell me: how should I go about obtaining the product I need? The product I need is: instant noodles, please provide guidance for me based on the scene in the image."
    response, history = model.chat(tokenizer, query=query,history=None)

    print(response)
    return None

def predict(path_to_adapter):
    test_json="/mnt/zy/projects/report/Annotation/ts.json"
    with open(test_json, "r") as f:
        test_data=json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(path_to_adapter, device_map="cuda", trust_remote_code=True).eval()

    predictions=[]
    adapter_name_suffix = os.path.basename(path_to_adapter).replace('output_qwen_', '')

    for item in test_data:
        item_id=item["id"]
        query=item["conversations"][0]["value"]
        print(query)
        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        item_pred = json.loads(json.dumps(item))
        item_pred["conversations"][1]["value"]=response
        predictions.append(item_pred)

    with open("/mnt/zy/projects/report/Annotation/pred_{}.json".format(adapter_name_suffix), "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

def icl(int_mark="int4"):
    test_json="/mnt/zy/projects/report/Annotation/ts.json"
    with open(test_json, "r") as f:
        test_data=json.load(f)

    if int_mark=="int4":
        model_iden="Qwen/Qwen-VL-Chat-Int4"
        tokenizer = AutoTokenizer.from_pretrained(model_iden, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_iden, device_map="cuda",
                                                         trust_remote_code=True).eval()
    elif int_mark=="half":
        model_iden="Qwen/Qwen-VL-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_iden, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_iden, device_map="cuda",
                                                         trust_remote_code=True).half().eval()
    else:
        print("Please make sure the int_mark is correct!")
        return



    predictions=[]

    for item in test_data:
        item_id=item["id"]
        query=item["conversations"][0]["value"]
        print(query)
        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        item_pred = json.loads(json.dumps(item))
        item_pred["conversations"][1]["value"]=response
        predictions.append(item_pred)

    with open("/mnt/zy/projects/report/Annotation/pred_icl_{}.json".format(int_mark), "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)



if __name__=="__main__":
    icl('int4')
    icl('half')
    # path_to_adapter="/mnt/zy/projects/report/Qwen-VL/output_qwen_ds"
    # predict(path_to_adapter)
    # path_to_adapter="/mnt/zy/projects/report/Qwen-VL/output_qwen_one_gpu"
    # predict(path_to_adapter)






#
# if __name__=='__main__':
#     after_ft_inference()
# #     #quantization_inference()

