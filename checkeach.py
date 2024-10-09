from tqdm import tqdm
import json
import os

def get_dic_from_json(jsonname):
    if not os.path.isfile(jsonname):
        print(f"{jsonname} is not exists.")
        return None
    with open(jsonname) as fp:
        new_dic = json.load(fp)
    return new_dic

if __name__ == "__main__":
    data_dir = "/mnt/mnt-quick/PixelAI/Public/Datasets/Voice/voice_drive/raw_data/multitalk/multitalk_dataset"
    facequality = "facequality/facequality/multitalk_dataset"
    syncnet = "syncnet/LSE/multitalk_dataset"
    emotion = "emotiondetect/emotion/multitalk_dataset"
    languages = os.listdir(data_dir)
    for language in tqdm(languages):
        print(f"{language}")
        jsonnames = os.listdir(f"{data_dir}/{language}")
        for jsonname in tqdm(jsonnames):
            # quality_dic = get_dic_from_json(f"{facequality}/{language}/{jsonname}.json")
            # emotion_dic = get_dic_from_json(f"{emotion}/{language}/{jsonname}.json")
            synchro_dic = get_dic_from_json(f"{syncnet}/{language}/{jsonname}.json")
            if synchro_dic is None:
                continue
            vid_ids = os.listdir(f"{data_dir}/{language}/{jsonname}/split_video_25fps_frame")
            for vid_id in vid_ids:
                try:
                    # quality = quality_dic[vid_id]["quality"]
                    # noface = quality_dic[vid_id]["noface_frame"]
                    # emo_ = emotion_dic[vid_id]["emotion"]
                    # credibility_ = emotion_dic[vid_id]["credibility"]
                    LSEC = synchro_dic[vid_id]["LSE-C"]
                    LSED = synchro_dic[vid_id]["LSE-D"]
                except:
                    print(f"{jsonname}/{vid_id} get wrong")
                    # continue
