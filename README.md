# pytorch-minist-ddp-on-sagemaker

This is a Pytorch distributed training demo code on Amazon SageMaker that support multi host with multi cards.


Instead of using *torch.distributed.launch* here we use *torch.multiprocessing* instead. 


For initialize the distributed environment, we use env as init_method.
```
dist.init_process_group(backend=args.backend, init_method='env://', rank=rank, world_size=world_size)
```
And set **MASTER_ADDR** and **MASTER_PORT** as well.
```
os.environ['MASTER_ADDR'] = master #args.hosts[0]
os.environ['MASTER_PORT'] = '23456'  
```

## Log

Remove PyTorch DP when use single host multi GPU, change to use DDP as well.

## Referecne
https://pytorch.org/docs/stable/distributed.html