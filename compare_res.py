import json
import re

def merge_res(pred_file, mark, gt_file="/mnt/zy/projects/report/Annotation/ts.json"):
    with open(pred_file,'r')as f:
        pred=json.load(f)
    with open(gt_file,'r') as f:
        gt=json.load(f)

    up_items=[]

    for i in range(len(pred)):
        pred_i=pred[i]
        gt_i=gt[i]

        #get id
        i=pred_i['id']

        #get prediciton and ground truth
        p=pred_i['conversations'][1]['value']
        g=gt_i['conversations'][1]['value']

        #get product name
        question=pred_i['conversations'][0]['value']
        pattern = r"The product I need is: (.*?),"
        match = re.search(pattern, question)
        product= match.group(1).strip()

        #First function: determine whether the model can see the product
        #score_vis=get_vis_score(p,product)

        #Second function: determine whether the model provide guidance to the user for getting the product,and to what degree the answer is correct
        #score_guide=get_guide_score(p,g)

        item_up={'id':i, 'product':product, 'prediction':p, 'ground truth':g}
        up_items.append(item_up)

    with open('/mnt/zy/projects/report/Annotation/res_{}.json'.format(mark), 'w') as f:
        json.dump(up_items, f, indent=4)

if __name__=="_main__":
    # pred_file = "/mnt/zy/projects/report/Annotation/pred_one_gpu.json"
    # merge_res(pred_file, mark="one_gpu")
    #
    # pred_file = "/mnt/zy/projects/report/Annotation/pred_ds.json"
    # merge_res(pred_file, mark="ds")

    pred_file = "/mnt/zy/projects/report/Annotation/pred_icl_int4.json"
    merge_res(pred_file, mark="int4")

    pred_file = "/mnt/zy/projects/report/Annotation/pred_icl_half.json"
    merge_res(pred_file, mark="half", gt_file="/mnt/zy/projects/report/Annotation/ts.json")







