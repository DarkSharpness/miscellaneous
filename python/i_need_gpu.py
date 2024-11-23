# this program may help you make others' processes OOM
import torch

MIN_MEMORY  = 1 << 20 # 1MB as the minimum memory
MAX_MEMORY  = 1 << 40 # 1TB as the maximum memory

lists = []

num_gpus = torch.cuda.device_count()
which = 0

mem_list = [1 << 30] * num_gpus

while True:
    current = mem_list[which]
    while True:
        assert isinstance(current, int)
        if current < MIN_MEMORY:
            current = MIN_MEMORY
        if current > MAX_MEMORY:
            current = MAX_MEMORY
        try:
            a = torch.zeros(current, dtype=torch.bool, device="cuda:0")
            lists.append(a)
            current = current * 2
        except torch.OutOfMemoryError:
            current = current // 2
            break
    mem_list[which] = current
    which = (which + 1) % num_gpus
