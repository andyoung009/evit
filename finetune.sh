# 获取当前日期和时间，并将其格式化为一个字符串，以便在脚本中使用。
now=$(date +"%Y%m%d_%H%M%S")
logdir=./train_log/exp_$now
datapath=" /data/ML_document/ImageNet/ILSVRC/Data/CLS-LOC/"
ckpt=deit_small_patch16_224-cd65a155.pth

echo "output dir: $logdir"

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env \
	main.py \
	--model deit_small_patch16_shrink_base \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--sched cosine \
	--lr 2e-5 \
	--min-lr 2e-6 \
	--weight-decay 1e-6 \
	--batch-size 256 \
	--shrink_start_epoch 0 \
	--warmup-epochs 0 \
	--shrink_epochs 0 \
	--epochs 30 \
	--dist-eval \
	--finetune $ckpt \
	--data-path $datapath \
	--output_dir $logdir

echo "output dir for the last exp: $logdir"\

# 这段代码是一个 Bash 脚本，主要用于启动一个分布式训练任务，并将日志输出到指定目录中。
# 具体来说，它做了以下事情：
# 1. 获取当前时间，并将其格式化为 `%Y%m%d_%H%M%S` 的形式，赋值给 `now` 变量。
# 2. 构造日志输出目录，将 `now` 变量插入到目录名中，赋值给 `logdir` 变量。
# 3. 设置数据集路径变量 `datapath` 和预训练模型的路径变量 `ckpt`。
# 4. 输出日志输出目录位置。
# 5. 使用 `torch.distributed.launch` 命令启动分布式训练任务。该命令使用 `main.py` 脚本作为入口，使用 8 个进程进行训练，启用环境变量，并传递一些参数给 `main.py` 脚本。
# 6. 输出上一次实验的日志输出目录位置。
# 其中，`main.py` 脚本是一个 PyTorch 分布式训练脚本，使用指定的模型在给定的数据集上进行训练，并将日志输出到指定的目录中。该脚本会解析传递给它的参数，并启动分布式训练任务。