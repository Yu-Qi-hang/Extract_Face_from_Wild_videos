'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: iic/emotion2vec_base_finetuned
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''
'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import glob
import random
import argparse
emos = {
    "0": "angry 生气",
    "1": "contempt 其他",
    "2": "disgusted 厌恶",
    "3": "fear 恐惧",
    "4": "happy 开心",
    "5": "neutral 中立",
    "6": "sad 难过",
    "7": "surprised 吃惊"
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/mnt-quick/PixelAI/Public/Datasets/Voice/voice_drive/train_data/MEAD")
    parser.add_argument("--speaker", type=str, default=None, required=True)
    parser.add_argument("--mode", type=str, default="utterance")
    args = parser.parse_args()
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_base_finetuned")  # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned
    speaker_root = os.path.join(args.data_root, args.speaker)
    all_wavs = glob.glob(os.path.join(speaker_root, "split_video_25fps_audio/*.wav"))
    wavs = all_wavs
    output_dir = os.path.join(speaker_root, "split_video_25fps_emotion")
    os.makedirs(output_dir,exist_ok=True)
    rec_result = inference_pipeline(wavs, output_dir=output_dir, granularity=args.mode)
# rec_result = inference_pipeline(wavs, output_dir="./outputs", granularity="utterance", extract_embedding=False)
# right = 0
# for choice in range(len(rec_result)):
#     scores_list = sorted(rec_result[choice]["scores"], reverse=True)
#     gt_emo = emos[os.path.basename(wavs[choice]).split("_")[0]]
#     print(gt_emo,end=">>")
#     for i in range(1):
#         res_emo = rec_result[choice]["labels"][rec_result[choice]["scores"].index(scores_list[i])]
#         print(res_emo, scores_list[i])
#         if res_emo[:2] in gt_emo:
#             right+=1
# print(f"accuracy:{right/len(wavs)}")
# print(len(rec_result[0]['feats']))