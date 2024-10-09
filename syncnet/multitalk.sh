for language in "$@"
do
echo processing $language
vids=$(ls multitalk_dataset/$language -r)
for vid in $vids
do
   python avsync_fast.py --data_dir tmp_dir/$language/$vid --video_dir multitalk_dataset/$language/$vid --min_track 10
   rm -r tmp_dir/$language/$vid &
done
done
