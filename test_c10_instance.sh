noise_type="instance" # pairflip symmetric instance manual
methods="mv rank1" # mv rank1 both
pre_type="image18 CLIP" # image18 image34 image50 ssl_c10 ssl_c100 CLIP clean_label_10
# pre_type="clean_label_100" # image18 image34 image50 ssl_c10 ssl_c100 CLIP clean_label_10
for PRE in $pre_type;
do
	for NOISE in $noise_type;
	do
		for METHOD in $methods;
		do
			echo "Running model $PRE, noise $NOISE, method $METHOD"
			CUDA_VISIBLE_DEVICES=0 python3  main_fast.py --dataset cifar10 --noise_type $NOISE --k 10 --pre_type $PRE --noise_rate 0.4 --num_epoch 21 --Tii_offset 1.0 --method $METHOD  > ./results/cifar10/c10_$METHOD\_$PRE\_$NOISE.log
		done
	done
done
