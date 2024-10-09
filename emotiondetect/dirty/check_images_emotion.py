# from transformers import pipeline
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
from tqdm import tqdm
import cv2
import numpy as py
import glob
import os
import json
import random

emo_MEAD = ["angry","contempt","disgusted","fearful","happy","neutral","sad","surprised"]
if __name__ == "__main__":
    pipe = pipeline(Tasks.facial_expression_recognition, 'damo/cv_vgg19_facial-expression-recognition_fer')
    # pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    data_root = '/mnt/mnt-quick/PixelAI/Public/Datasets/Voice/voice_drive/train_data/MEAD'
    speakers = os.listdir(data_root)
    total_num = 0
    total_right = 0
    data_dic = {}
    for speaker in speakers:
        src_img_root = f'/mnt/mnt-quick/PixelAI/Public/Datasets/Voice/voice_drive/train_data/MEAD/{speaker}/split_video_25fps_frame'
        videos = os.listdir(src_img_root)
        speaker_dic = {}
        speaker_num = 0
        speaker_right = 0
        for video in tqdm(videos):
            gt_emo = emo_MEAD[int(video.split("_")[0])]
            if gt_emo == "contempt":
                continue
            speaker_num += 1
            src_frame_path = os.path.join(src_img_root, video)
            src_img_paths_all = glob.glob(os.path.join(src_frame_path,"*.png"))
            src_img_paths = random.choices(src_img_paths_all, k=80) if len(src_img_paths_all) > 80 else src_img_paths_all
            image_emos = {"sad":0, "disgust":0, "angry":0, "neutral":0, "fear":0, "surprise":0, "happy":0}
            for src_img_path in src_img_paths:
                image_emo = pipe(src_img_path)
                # for emo in image_emo:
                #     image_emos[emo["label"]] += emo["score"]
                for i in range(7):
                    image_emos[image_emo["labels"][i].lower()] += image_emo["scores"][i]
            image_emos["disgusted"] = image_emos["disgust"]
            image_emos["fearful"] = image_emos["fear"]
            image_emos["surprised"] = image_emos["surprise"]
            image_emos.pop("disgust")
            image_emos.pop("fear")
            image_emos.pop("surprise")
            image_keys = list(image_emos.keys())
            image_values = list(image_emos.values())
            det_emo = image_keys[image_values.index(max(image_values))]
            speaker_dic[video] = {}
            speaker_dic[video]["emo_label"] = det_emo
            speaker_dic[video]["emo_id"] = emo_MEAD.index(det_emo)
            if det_emo == gt_emo:
                speaker_right += 1
        total_num += speaker_num
        total_right += speaker_right
        data_dic[speaker] = speaker_dic
        print(f'The accuracy of {speaker} is {speaker_right/speaker_num}')
        with open(os.path.basename(data_root)+"_emo_det.json","w") as fp:
            json.dump(data_dic, fp, indent=4) 
    print(f'the accuracy of above if {total_right/total_num}')
    # with open(os.path.basename(data_root)+"_emo_det.json","w") as fp:
    #     json.dump(data_dic, fp, indent=4)