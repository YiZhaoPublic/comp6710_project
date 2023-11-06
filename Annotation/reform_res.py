import json

# mark = "ds"
# mark = "one_gpu"
import pdb

def reform_res(mark):
    res_file = "res_{}.json".format(mark)
    test_file = "ts.json"
    feedback_file = "res_{}_feedback_gpt4.json".format(mark)

    with open(res_file, "r") as f:
        data_res = json.load(f)

    with open(feedback_file, "r") as f:
        data_fb = json.load(f)

    with open(test_file, "r") as f:
        data_test = json.load(f)

    reform_items = []

    score_vis_list=[]
    score_guidance_list=[]

    #pdb.set_trace()
    for i in range(10):
        item_res_i = data_res[i]

        product_i = item_res_i['product']
        prediction_i = item_res_i['prediction']
        gt_i = item_res_i['ground truth']

        item_fb_i = data_fb[i]
        score_vis_i = item_fb_i['score_vis']
        score_guidance_i = item_fb_i['score_guidance']
        score_vis_list.append(score_vis_i)
        score_guidance_list.append(score_guidance_i)

        item_gt_i = data_test[i]
        id_i = item_gt_i['id']
        img_id_i = int(id_i.replace('identity_', ''))
        img_path_i = "/mnt/zy/projects/report/resized_images_1024_768/{}.jpg".format(img_id_i)

        item_up_i = {"img_id": img_id_i, "img_path": img_path_i, "product": product_i, "prediction_i": prediction_i,
                     "ground truth": gt_i, "score_vis": score_vis_i, "score_guidance": score_guidance_i}

        reform_items.append(item_up_i)


    with open("reformed_res_{}.json".format(mark), "w") as f:
        json.dump(reform_items, f, indent=4)

    return score_vis_list, score_guidance_list


if __name__=="__main__":
    for mark in ['ds','one_gpu','int4','half']:
        a,b=reform_res(mark)
        print(mark)
        print('score_vis_list',a,sum(a)/len(a))
        print('score_guidance_list',b,sum(b)/len(b))