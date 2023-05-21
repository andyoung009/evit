now=$(date +"%Y%m%d_%H%M%S")
logdir=./train_log/exp_$now
datapath="/data/ML_document/ImageNet/ILSVRC/Data/CLS-LOC/"

echo "output dir: $logdir"

python3 -m torch.distributed.launch --nproc_per_node=2 --use_env \
	main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--batch-size 128 \
	--warmup-epochs 5 \
	--shrink_start_epoch 10 \
	--shrink_epochs 100 \
	--epochs 300 \
	--dist-eval \
	--data-path $datapath \
	--output_dir $logdir

echo "output dir for the last exp: $logdir"\

# 在这个命令中，`--use_env`选项是`torch.distributed.launch`模块提供的一个选项，用于指定是否使用环境变量来设置分布式训练的参数。
# 当你使用`torch.distributed.launch`模块来启动一个分布式训练任务时，可以使用`--use_env`选项来指定是否使用环境变量来设置分布式训练的参数。
# 具体来说，`--use_env`选项会将当前进程的环境变量传递给所有启动的训练进程，以便它们可以共享相同的环境变量。
# 这个选项通常用于在分布式训练中设置一些共享的参数，例如数据路径、日志路径等。
# 在这个命令中，`--use_env`选项被用于启动一个名为`main.py`的Python脚本，该脚本使用了PyTorch的分布式训练功能。
# 这个命令会启动两个训练进程，每个进程使用一个GPU来训练模型。`--data-path`和`--output_dir`选项用于设置数据路径和日志路径，
# 它们的值是通过环境变量`$datapath`和`$logdir`来传递的。因此，`--use_env`选项确保了这些环境变量在所有训练进程中都是可用的。