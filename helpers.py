import math
import time
import torch
from torchprofile import profile_macs


def adjust_keep_rate(iters, epoch, warmup_epochs, total_epochs,
                       ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate


def speed_test(model, ntest=100, batchsize=64, x=None, **kwargs):
    if x is None:
        img_size = model.img_size
        x = torch.rand(batchsize, 3, *img_size).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    start = time.time()
    for i in range(ntest):
        model(x, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse
    # speed = torch.tensor(speed, device=x.device)
    # torch.distributed.broadcast(speed, src=0, async_op=False)
    # speed = speed.item()
    return speed


def get_macs(model, x=None):
    model.eval()
    if x is None:
        img_size = model.img_size
        x = torch.rand(1, 3, *img_size).cuda()
    macs = profile_macs(model, x)
    return macs


# 计算其他非topk的token融合的代码
def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    # 这行代码将 dims 中的最后一维替换为 -1。这是因为在计算补集时，我们不知道补集的大小，因此我们需要将其保留为未知的 -1，以便在计算张量形状时使用。
    # 在这里，dims 是输入张量 idx 的形状。由于我们只需要替换最后一维，所以代码使用了切片操作 dims[:-1] 来获取除了最后一维之外的所有维度，
    # 并将其与一个只包含 -1 的元组连接起来来创建新的形状。
    dims = dims[:-1] + (-1, )
    # 扩充a的维度信息，使得a的维度信息和idx的维度信息一致
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    # 把a每个维度的尺寸按照dims来扩展，因为之前dims最后一个维度信息换成了-1，见‘dims = dims[:-1] + (-1, )’
    # a的shape比idx在最后一个维度大，多包含了非topk的tokens
    a = a.expand(*dims)
    # 这行代码使用 PyTorch 的 `scatter` 函数，将 `a` 中的值沿着最后一维度（即 `dim` 的维度）根据 `idx` 张量中的索引进行覆盖。
    # 具体来说，`scatter` 函数的第一个参数是要被覆盖的张量，第二个参数是指定索引的维度，这里是 `-1` 表示最后一维，第三个参数是一个索引张量，用于指定要覆盖的位置
    # 第四个参数是将要覆盖的值。在这里，我们将 `0` 作为要覆盖的值，因为我们想要在 `a` 张量中保留补集的位置为 `0`，而原来的元素都被覆盖为 `0`。
    # 因此，`masked` 张量中，与 `idx` 张量中指定的位置对应的元素被设置为 `0`，而其他位置的元素保持不变。
    masked = torch.scatter(a, -1, idx, 0)
    # 这三行代码用于计算补集，并将其排列成与输入张量 `idx` 相同的形状。
    # 第一行代码使用 PyTorch 的 `sort` 函数对张量 `masked` 沿着最后一维（即 `idx` 的维度）进行排序，并将排序后的结果存储在 `compl` 张量中。
    # 因为我们想要计算补集，所以我们需要按升序对张量进行排序。第二个返回值 `_` 是排序后的原始索引，因为我们在这里不需要使用它，所以用占位符 `_` 来接收。
    # 第二行代码使用 `permute` 函数重排 `compl` 张量的维度，以最终得到与输入张量 `idx` 相同的形状。具体来说，我们首先将最后一维度移动到第一维度，
    # 然后将除了第一维度之外的所有维度按原始顺序排列。
    # 第三行代码使用切片操作 `(n_idx:, )` 来获取 `compl` 张量中的补集部分，并使用 `permute` 函数将其重排成与输入张量 `idx` 相同的形状。在此之前，
    # 我们需要将 `compl` 张量的第一维（即之前移动到第一维的最后一维）移回到最后一维，以便在使用 `permute` 函数时得到正确的形状。
    # 具体来说，我们先将维度从 `(0, 1, ..., ndim-1)` 转换为 `(*range(1, ndim), 0)`，然后使用 `permute` 函数重排维度。
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    # 切片后形成的coml时非topk的tokens的集合
    return compl
