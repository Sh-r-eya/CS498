###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #  
    for i in range(world - 1):
        send_idx = (rank - i) % world
        recv_idx = (rank - i - 1) % world

        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(tmp)

        send_req = dist.isend(send_buf, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        chunks[recv_idx].add_(recv_buf)
    return
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #
    for i in range(world - 1):
        send_idx = (rank - i - 1) % world
        recv_idx = (rank - i - 2) % world

        send_buf = chunks[send_idx].contiguous()
        recv_buf = torch.empty_like(tmp)

        send_req = dist.isend(send_buf, dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        chunks[recv_idx].copy_(recv_buf)                                                                #
    return

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    # 
    padded_len = chunk * world
    if padded_len > n:
        pad = torch.zeros(padded_len - n, dtype=flat.dtype, device=flat.device)
        padded_flat = torch.cat([flat, pad], dim=0)
    else:
        padded_flat = flat                                                               #
    #So, fill zeros at the end of flat to generate padded_flat
    #padded_flat = None # modify this line and fill correct value into padded_flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]
    tmp = torch.empty_like(chunks[0])
    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    reduce_scatter(chunks, tmp, world, rank, left, right)                                                                  #
    all_gather(chunks, tmp, chunks[rank], world, rank, left, right)
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    
    # stitch & unpad  
    flat_reduced = torch.cat(chunks, dim=0)[:n]
    flat_reduced.div_(world)
    tensor.view(-1).copy_(flat_reduced)
    return
