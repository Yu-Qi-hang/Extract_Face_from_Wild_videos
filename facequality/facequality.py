from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from tqdm import tqdm
import numpy as np
import random
import argparse
import json
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="", help="data root of videos to be detected")
    args = parser.parse_args()

    face_quality_assessment_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa')

    save_root = "/mnt/mnt-quick/PixelAI/Private/qianyu/code2/dataclean/facequality/facequality/multitalk_dataset"
    save_dir = f"{save_root}/{args.language}"
    language_dir = f"multitalk_dataset/{args.language}"
    os.makedirs(save_dir,exist_ok=True)
    vid_ids = os.listdir(language_dir)
    for vid_id in tqdm(vid_ids):
        face_scores = {}
        # wrong_clips = []
        json_path = f"{save_dir}/{vid_id}.json"
        if os.path.isfile(json_path):
            with open(json_path) as fp:
                face_scores = json.load(fp)
        vid_clips = os.listdir(f"{language_dir}/{vid_id}/split_video_25fps_frame")
        for vid_clip in tqdm(vid_clips):
            if vid_clip in face_scores:
                continue
            clip_score = []
            noface_frame = False
            images_all = glob.glob(f"{language_dir}/{vid_id}/split_video_25fps_frame/{vid_clip}/*.png")
            images = random.sample(images_all, min(len(images_all),10))
            images.append(images_all[-1])
            images.append(images_all[0])
            for image in images:
                try:
                    frame_score = face_quality_assessment_func(image)[OutputKeys.SCORES][0]
                except:
                    noface_frame = True
                    # if vid_clip not in wrong_clips:
                        # wrong_clips.append(vid_clip)
                    frame_score = 0
                clip_score.append(frame_score)
            face_scores[vid_clip] = {"quality":sum(clip_score)/len(clip_score), "noface_frame":noface_frame}
        # with open(json_path,"w") as fp, open(json_path.replace(".json",".txt"),"w")as fr:
        #     json.dump(face_scores, fp, indent=4)
        #     fr.write("\n".join(wrong_clips))
        with open(json_path,"w") as fp:
            json.dump(face_scores, fp, indent=4)

