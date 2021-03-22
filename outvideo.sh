for ((i=2;i<82;i++))
do
for file in iv/Clip$i.mp4
do
	echo $file
	outfile=ov/Clip$i"_TrackNet.mp4"
	echo $outfile
	python  predict_video.py  --save_weights_path=weights/model.0 --input_video_path=$file --output_video_path=$outfile --n_classes=256
done
done

