from tqdm import tqdm
import json
import os

def get_dic_from_json(jsonname):
    with open(jsonname) as fp:
        new_dic = json.load(fp)
    return new_dic

if __name__ == "__main__":
    facequality = "facequality/facequality/multitalk_dataset"
    syncnet = "syncnet/LSE/multitalk_dataset"
    emotion = "emotiondetect/emotion/multitalk_dataset"
    save_dir = "filters/q_0_0_c_0_0_d_20_0_e_0_0"
    languages = os.listdir(facequality)
    languages = ['croatian']
    for language in tqdm(languages):
        os.makedirs(f"{save_dir}/{language}",exist_ok=True)
        jsonnames = os.listdir(f"{facequality}/{language}")
        for jsonname in tqdm(jsonnames):
            quality_dic = get_dic_from_json(f"{facequality}/{language}/{jsonname}")
            synchro_dic = get_dic_from_json(f"{syncnet}/{language}/{jsonname}")
            emotion_dic = get_dic_from_json(f"{emotion}/{language}/{jsonname}")
            new_dic = {}
            for vid_id, quality in quality_dic.items():
                new_dic[vid_id] = {}
                new_dic[vid_id]["quality"] = quality["quality"]
                new_dic[vid_id]["noface_frame"] = quality["noface_frame"]
                new_dic[vid_id]["emotion"] = emotion_dic[vid_id]["emotion"]
                new_dic[vid_id]["credibility"] = emotion_dic[vid_id]["credibility"]
                new_dic[vid_id]["LSE-C"] = synchro_dic[vid_id]["LSE-C"]
                new_dic[vid_id]["LSE-D"] = synchro_dic[vid_id]["LSE-D"]
            with open(f"{save_dir}/{language}/{jsonname}","w") as fp:
                json.dump(new_dic,fp,indent=4)
