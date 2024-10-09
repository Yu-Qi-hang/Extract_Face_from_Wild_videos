for language in "$@"
do
echo processing $language
mkdir -p emotion/multitalk_dataset/$language
vids=$(ls multitalk_dataset/$language)
for vid in $vids
do
   python extract_emotion_video.py --output emotion/multitalk_dataset/$language --data_root /mnt/mnt-quick/PixelAI/Public/Datasets/Voice/voice_drive/raw_data/multitalk/multitalk_dataset/$language/$vid
done
done
