speakers=$(ls $1)
for speaker in $speakers
do
python extract_emotion2vec.py --data_root $1 --speaker $speaker
done