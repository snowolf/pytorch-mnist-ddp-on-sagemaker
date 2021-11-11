# pytorch-minist-ddp-on-sagemaker

This is from https://github.com/muhyun/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/pytorch_mnist, it fixed the DDP issue, so it could really use multi host with multi card.

Instead of using Pytroch distributed training launcher, here we use multiprocess instead. 

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