# from transformers import pipeline as tp
from modelscope.pipelines import pipeline as mp
from modelscope.utils.constant import Tasks
from multiprocessing import Pool, Manager
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import cv2
import glob
import json
import random
import argparse
import subprocess
import logging
class VideoEmo():
    def __init__(self, data_root="", video="", output="", perframe=1, verbose=False):
        super(VideoEmo, self).__init__()
        # build detectation models
        # self.image2emo = tp("image-classification", model="dima806/facial_emotions_image_detection",framework="pt")
        # self.image2emo = mp(task=Tasks.facial_expression_recognition, model='damo/cv_vgg19_facial-expression-recognition_fer')
        # self.audio2emo = mp(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")  # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned
        # build dirs
        self.loaded = False
        self.verbose = verbose
        self.data_root = data_root[:-1] if data_root[-1] == '/' else data_root
        self.output = self.data_root + "_verbose" if len(output)==0 else output
        os.makedirs(self.output, exist_ok=True)
        # obtain videos
        self.videos = [os.path.join(self.data_root,video)] if len(video) else glob.glob(os.path.join(self.data_root,"split_video_25fps","*.mp4"))
        # define emotions
        self.perframe = perframe
        self.hot_id = {'angry':0b1, 'disgusted':0b10, 'fearful':0b100, 'happy':0b1000, 'neutral':0b10000, 'sad':0b100000, 'surprised':0b1000000}
        # image_emos = {"sad":0, "disgust":0, "angry":0, "neutral":0, "fear":0, "surprise":0, "happy":0}

    def load_model(self):
        # build detectation models
        # self.image2emo = tp("image-classification", model="dima806/facial_emotions_image_detection",framework="pt")
        self.image2emo = mp(task=Tasks.facial_expression_recognition, model='damo/cv_vgg19_facial-expression-recognition_fer')
        self.audio2emo = mp(task=Tasks.emotion_recognition, model="iic/emotion2vec_plus_large")  # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned
        self.loaded = True
        return

    def __get_audio_emotion(self, video, output_dir):
        # audio_path = os.path.join(output_dir, "audio.wav")
        audio_path = video.replace(".mp4",".wav").replace("split_video_25fps","split_video_25fps_audio")
        # subprocess.call(f'ffmpeg -y -i {video} -vn -acodec libmp3lame -ab 160k -loglevel error {audio_path}', shell=True)
        audio_emo = self.audio2emo(audio_path, granularity="utterance", extract_embedding=False)
        audio_emo = audio_emo[0]
        audio_emos = {"angry":0, "disgusted":0, "fearful":0, "happy":0, "neutral":0, "other":0, "sad":0, "surprised":0, "unknown":0}
        for idx, keys in enumerate(audio_emo["labels"]):
            if keys == '<unk>':
                audio_emos["unknown"] += audio_emo["scores"][idx]
            else:
                audio_emos[keys.split('/')[-1]] += audio_emo["scores"][idx]
        audio_emos.pop("other")
        audio_emos.pop("unknown")
        # if not self.verbose:
        #     subprocess.call(f'rm {audio_path}',shell=True)
        return audio_emos

    def __get_image_emotion3(self, video, output_dir):
        frame_dir = video.replace(".mp4","").replace("split_video_25fps","split_video_25fps_frame")
        ## 人脸情绪识别
        image_emos = {"sad":0, "disgust":0, "angry":0, "neutral":0, "fear":0, "surprise":0, "happy":0}
        image_names_all = glob.glob(os.path.join(frame_dir,"*.png"))
        image_names = random.sample(image_names_all, min(len(image_names_all),30))
        for image_name in image_names:
            frame_PIL = Image.open(image_name)
            image_emo = self.image2emo(frame_PIL)
            for i in range(7):
                try:
                    image_emos[image_emo["labels"][i].lower()] += image_emo["scores"][i]
                except TypeError:
                    print(image_emo)
        image_emos["disgusted"] = image_emos["disgust"]
        image_emos["fearful"] = image_emos["fear"]
        image_emos["surprised"] = image_emos["surprise"]
        image_emos.pop("disgust")
        image_emos.pop("fear")
        image_emos.pop("surprise")
        return image_emos

    def __get_image_emotion2(self, video, output_dir):
        frame_dir = os.path.join(output_dir, "frames") # 图像位置
        os.makedirs(frame_dir, exist_ok=True)
        ## 人脸情绪识别
        cap = cv2.VideoCapture(video)
        frame_index = 0 
        image_emos = {"sad":0, "disgust":0, "angry":0, "neutral":0, "fear":0, "surprise":0, "happy":0}
        while True:
            success, frame = cap.read()  # 读取一帧
            if not success:
                break  # 如果没有帧了，就退出循环
            if frame_index%self.perframe == 0:
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_PIL = Image.fromarray(frame_RGB)
                image_emo = self.image2emo(frame_PIL)
                for i in range(7):
                    image_emos[image_emo["labels"][i].lower()] += image_emo["scores"][i]
            # image_emos.append(image_emo)
            if self.verbose:
                filename = f"{frame_dir}/{frame_index:06d}.png"
                cv2.imwrite(filename, frame)  # 保存帧为图片文件
            frame_index += 1  # 更新帧索引
        cap.release()  # 释放视频对象
        image_emos["disgusted"] = image_emos["disgust"]
        image_emos["fearful"] = image_emos["fear"]
        image_emos["surprised"] = image_emos["surprise"]
        image_emos.pop("disgust")
        image_emos.pop("fear")
        image_emos.pop("surprise")
        if not self.verbose:
            subprocess.call(f'rm -r {frame_dir}', shell=True)
        return image_emos

    def __get_image_emotion(self, video, output_dir):
        frame_dir = os.path.join(output_dir, "frames") # 图像位置
        os.makedirs(frame_dir, exist_ok=True)
        ## 人脸情绪识别
        cap = cv2.VideoCapture(video)
        frame_index = 0 
        image_emos = {"sad":0, "disgust":0, "angry":0, "neutral":0, "fear":0, "surprise":0, "happy":0}
        while True:
            success, frame = cap.read()  # 读取一帧
            if not success:
                break  # 如果没有帧了，就退出循环
            if frame_index%self.perframe == 0:
                frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_PIL = Image.fromarray(frame_RGB)
                image_emo = self.image2emo(frame_PIL)
                for emo in image_emo:
                    image_emos[emo["label"]] += emo["score"]
            # image_emos.append(image_emo)
            if self.verbose:
                filename = f"{frame_dir}/{frame_index:06d}.png"
                cv2.imwrite(filename, frame)  # 保存帧为图片文件
            frame_index += 1  # 更新帧索引
        cap.release()  # 释放视频对象
        image_emos["disgusted"] = image_emos["disgust"]
        image_emos["fearful"] = image_emos["fear"]
        image_emos["surprised"] = image_emos["surprise"]
        image_emos.pop("disgust")
        image_emos.pop("fear")
        image_emos.pop("surprise")
        if not self.verbose:
            subprocess.call(f'rm -r {frame_dir}', shell=True)
        return image_emos

    def __get_video_emotion(self, video, shared_dict):
        video_dir = os.path.join(self.output, os.path.basename(video)[:-4])
        # os.makedirs(video_dir, exist_ok=True)
        # json_path = os.path.join(self.output,os.path.basename(self.data_root)+".json")
        audio_emos = self.__get_audio_emotion(video,video_dir)
        image_emos = self.__get_image_emotion3(video,video_dir)
        # if not self.verbose:
        #     subprocess.call(f'rm -r {video_dir}', shell=True)
        ## 判断情绪是否一致
        audio_hot = 0b0
        image_hot = 0b0
        audio_keys = list(audio_emos.keys())
        audio_values = list(audio_emos.values())
        audio_values = self.__normallist(audio_values)
        scores_list = sorted(audio_values, reverse=True)
        for i in range(3):
            if scores_list[i] > 0.1:
                audio_hot += self.hot_id[audio_keys[audio_values.index(scores_list[i])]]
        image_keys = list(image_emos.keys())
        image_values = list(image_emos.values())
        image_values = self.__normallist(image_values)
        scores_list = sorted(image_values, reverse=True)
        for i in range(3):
            if scores_list[i] > 0.1:
                image_hot += self.hot_id[image_keys[image_values.index(scores_list[i])]]
        ## 输出视频对应的情绪
        rec_emos = []
        emo_hot = audio_hot&image_hot
        for emo_label, emo_id in self.hot_id.items():
            if emo_hot & emo_id:
                rec_emos.append(emo_label)
        if len(rec_emos)==0:
            final_emo = "neutral"
        elif len(rec_emos)==1:
            final_emo = rec_emos[0]
        else:
            scores = []
            for rec_emo in rec_emos:
                scores.append(audio_values[audio_keys.index(rec_emo)]) # audio first
                # scores.append(image_values[image_keys.index(rec_emo)])
                # scores.append(image_values[image_keys.index(rec_emo)]+audio_values[audio_keys.index(rec_emo)])
            final_emo = rec_emos[scores.index(max(scores))]
        emo_cred = max(audio_values[audio_keys.index(final_emo)],image_values[image_keys.index(final_emo)])
        shared_dict[os.path.basename(video)[:-4]] = {"emotion":final_emo,"credibility":emo_cred}

        return

    def get_videos_emotion(self):
        json_path = os.path.join(self.output,os.path.basename(self.data_root)+".json")
        manager = Manager()
        if os.path.isfile(json_path):
            shared_dict = manager.dict(json.load(open(json_path)))
        else:
            shared_dict = manager.dict()
        print(dict(shared_dict))
        sbar = tqdm(total=len(self.videos))
        update=lambda *args: sbar.update()
        for idx, video in enumerate(self.videos):
            print(video)
            if os.path.basename(video)[:-4] not in shared_dict:
                if not self.loaded:
                    self.load_model()
                self.__get_video_emotion(video,shared_dict)
            if idx % 20 == 0:##避免出现中断
                with open(json_path, "w") as fp:
                    json.dump(dict(shared_dict), fp, indent=4)
            sbar.update()
        sbar.close()
        video_emo_pairs = dict(shared_dict)
        manager.shutdown()
        return video_emo_pairs

    def get_emotions(self):
        print(self.hot_id.keys())

    def __normallist(self,mylist):
        total = sum(mylist)
        try:
            newlist = [x/total for x in mylist]
        except:
            newlist = mylist
        return newlist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="", help="data root of videos to be detected")
    parser.add_argument("--video", type=str, default="", help="the video name, none means all videos")
    parser.add_argument("--output", type=str, default="", help="the output dir")
    parser.add_argument("--perframe", type=int, default=1)
    parser.add_argument("--verbose", type=bool, default=False, help="whether store frames and audio")
    args = parser.parse_args()
    # logging.basicConfig(level=logging.ERROR) # 设置日志级别
    video_detectation = VideoEmo(data_root=args.data_root,video=args.video,output=args.output,perframe=args.perframe,verbose=args.verbose)
    video_emo_pairs = video_detectation.get_videos_emotion()
    with open(os.path.join(args.output,os.path.basename(args.data_root)+".json"), "w") as fp:
        json.dump(video_emo_pairs, fp, indent=4)
    print(f'save {os.path.join(args.output,os.path.basename(args.data_root)+".json")}')