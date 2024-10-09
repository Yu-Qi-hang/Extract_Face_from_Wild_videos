from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks
import numpy as np

fer = pipeline(Tasks.facial_expression_recognition, 'damo/cv_vgg19_facial-expression-recognition_fer')
# img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/facial_expression_recognition.jpg'
img_path = '/mnt/mnt-quick/PixelAI/Public/Datasets/Voice/voice_drive/train_data/MEAD/M003/split_video_25fps_frame/0_level_3_011/000000.png'
ret = fer(img_path)
# label_idx = np.array(ret['scores']).argmax()
# label = ret['labels'][label_idx]
print(f'facial expression : {ret}.')