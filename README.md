# Device Benchmarks

Benchmarks of different devices I have come across. This repo is migrated from this gist here: https://gist.github.com/chsasank/407df67ac0c848d6259f0340887648a9#file-benchmark-py

I will maintain interesting benchmarks of different devices of I have come across.

## Matrix Multiplication FLOPS and BW

I have written a quick script in PyTorch to benchmark GPUs and CPUs. I use fp32 matrix multiplication to measure FLOPs (floating point operations per second). I copy a large tensor to measure bandwidth. These two are the most important metrics for LLM inference. Read [this blog](https://chsasank.com/llm-system-design.html) for more details on this.


Here's an example run:

```
(ai) ➜  device-benchmarks git:(main) ✗ python benchmark.py --device='mps' --num_trails=10 --dtype=float32
Device "mps" torch.float32 matmul compute benchmark: 
Tensor size 2,048, Elapsed time 2.3250e-03s, 7.3891 TOPs.
Tensor size 2,435, Elapsed time 4.1079e-03s, 7.0293 TOPs.
Tensor size 2,896, Elapsed time 6.8624e-03s, 7.0787 TOPs.
Tensor size 3,444, Elapsed time 1.0083e-02s, 8.1028 TOPs.
Tensor size 4,096, Elapsed time 1.6305e-02s, 8.4293 TOPs.
Tensor size 4,870, Elapsed time 3.3050e-02s, 6.9895 TOPs.
Tensor size 5,792, Elapsed time 4.9070e-02s, 7.9195 TOPs.
Tensor size 6,888, Elapsed time 9.2078e-02s, 7.0983 TOPs.
Tensor size 8,192, Elapsed time 1.3304e-01s, 8.2648 TOPs.
Device "mps" yields 7.5890 TOPs on torch.float32 (min: 6.9895 TOPs, max: 8.4293 TOPs)

Device "mps" memory bandwidth benchmark: 
Copy size 0.0671 GB, Elapsed time 5.7410e-04s, Bandwidth 233.7899 GB/s
Copy size 0.1342 GB, Elapsed time 9.4940e-04s, Bandwidth 282.7410 GB/s
Copy size 0.2684 GB, Elapsed time 1.6748e-03s, Bandwidth 320.5623 GB/s
Copy size 0.5369 GB, Elapsed time 3.2969e-03s, Bandwidth 325.6773 GB/s
Copy size 1.0737 GB, Elapsed time 6.0067e-03s, Bandwidth 357.5142 GB/s
Copy size 2.1475 GB, Elapsed time 1.1745e-02s, Bandwidth 365.6863 GB/s
Copy size 4.2950 GB, Elapsed time 2.3810e-02s, Bandwidth 360.7683 GB/s
Device "mps" yields 320.9627 GB/s (min: 233.7899 GB/s, max: 365.6863 GB/s)

```

Some useful commands:

```
# for apple gpu
python benchmark.py --device mps --dtype float32

# for intel gpus with int8
python benchmark.py --device xpu --dtype int8

# for nvidia gpus with bfloat16
python benchmark.py --device cuda --dtype bfloat16
```


Here's a summary of the data I have collected for different devices

| Device | Device Type | TFLOPs (FP32) | TFLOPs (FP16)| TFLOPs (BF16) | TOPS (INT8) | Memory Bandwidth (GB/s) |
|---|---|---|---|---|---|---|
| Apple M1 CPU | CPU | 0.8 |  |  |  |  | 46 |
| Apple M1 GPU | GPU | 1.4 |  |  |  |  | 56 |
| Apple M1 Pro CPU 10-core | CPU | 0.3 |  |  | 0.008  | 96 |
| Apple M1 Pro GPU 16-core | GPU | 3.7 | 4.3 |  |  | 176 |
| Apple M2 CPU | CPU | 1 |    ||  | 60 |
| Apple M2 GPU | GPU | 2 |  | NA | NA | 90 |
| Apple M2 Ultra CPU | CPU | 4 |  |  |  | 311 |
| Apple M2 Ultra GPU (76 Core) | GPU | 20 |  |  |  | 636 |
| Apple M3 Max GPU (40 Core) | GPU | 11.4 |  |  |  | 318 |
| Apple M4 Max CPU (16C,40C,128G) | CPU | 2.85 |  |  |  | 336 |
| Apple M4 Max GPU (16C,40C,128G) | GPU | 13.27 | 13.29 |  |  | 396 |
| SteamDeck CPU | CPU | 0.17 | 0.002 | 0.002 | 0.05 | 20 |
| SteamDeck GPU | GPU | 1.22 | 2.2 | 0.5 | NA | 69 |
| Samsung Exynos 2100 | CPU | 0.1 |  |  |  | 16 |
| AMD Ryzen 5 3600 | CPU | 0.36 |  |  |  | 14 |
| AMD Ryzen 5 4600HS | CPU | 0.4 |  |  |  | 22 |
| AMD Ryzen 9 5900X | CPU | 1.3 |  |  |  | 29 |
| AMD Ryzen 9 7950X | CPU | 1.1 |  |  |  | 28 |
| AMD Ryzen Threadripper 3960X 24-Cores | CPU | 1.4 |  |  |  | 44 |
| AMD Ryzen Threadripper PRO 5975WX | CPU | 2.16 |  |  |  | 62 |
| AMD Epyc 7763 Engineering Sample | CPU | 3.2 |  |  |  | 115 |
| AMD Epyc 7262 | CPU | 0.5 |  |  |  | 80 |
| Intel i5-12400 | CPU | 0.7 |  | 0.003 | 0.05 | 26 |
| Intel i7-8559U | CPU | 0.2 |  |  |  | 10 |
| Intel i7-8750H | CPU | 0.5 |  |  |  | 15 |
| Intel i7-1360P | CPU | 0.4 |  | 0.003 | 0.06 | 24 |
| Intel i9-13900K (WSL2) | CPU | 1.2 |  |  |  | 49 |
| Intel i9-12950HX | CPU | 0.6 | | | | 40 |
| Intel Xeon Silver 4116 | CPU | 0.5 |  |  |  | 20 |
| Intel Xeon Gold 6230 | CPU | 1.9 | NA | 0.61 | 0.014 | 17.5 |
| Intel Xeon Gold 6330 | CPU | 5.7 | NA | 0.75 | 0.02 | 81 |
| Intel Xeon Platinum 8358 | CPU | 3.5 |  | 0.96 | 0.029  | 96 |
| Intel Xeon Platinum 8358 | CPU | 5.6 | NA | 14 | 0.04 | 137 |
| AMD 7900 XTX | GPU | 26 | 101 | 104 | NA | 792 |
| Intel Arc 770 16GB | GPU | 15 | 86 | 90 | 174 | 452 |
| Intel Arc 370m | GPU | 4 |  | 15 | 35 | 93 |
| Intel Data Center GPU Max 1100 | GPU | 21 | 140 | 140 | 221 | 781 |
| Nvidia T4 | GPU | 4 | 25 | 2.25 | NA | 240 |
| Nvidia L4 | GPU | 12 | 65 | 66 | NA | 235 |
| Nvidia V100 32GB | GPU | 13 | 84 | 9.4 | NA  | 766 |
| Nvidia A10 24GB | GPU | 14 | 54 | 56 | NA | 469 |
| Nvidia RTX 4000 Ada 20GB | GPU | 16 | 78 | 79 | NA | 300 |
| Nvidia A100 80GB | GPU | 19 | 189 | 237 | NA | 1490 |
| Nvidia H100-PCIe 80GB | GPU | 38 | 435 | 449 | NA  | 1630 |
| Nvidia 1050 Ti Mobile | GPU | 1.8 |1.5  | 1 | NA | 97 |
| Nvidia 1060 Ti Mobile | GPU | 3.8 | 17.6  | 2.18 | NA | 222 |
| Nvidia 1650 Ti Mobile | GPU | 3 |  | 1.8 | NA | 172 |
| Nvidia 2070S | GPU | 8 | 37 | 5 | NA | 831 |
| Nvidia 3060 12 GB | GPU | 7.4 | 26 | 26 | NA | 330 |
| Nvidia 3080 Ti Mobile | GPU | 13 | 46 | 45 | NA | 475 |
| Nvidia 3090 | GPU | 27 |  |  |  | 831 |
| Nvidia 4060ti | GPU | 12 | 42 | 46 | NA | 234 |
| Nvidia 4070 Super | GPU | 23 |  |  |  | 411 |
| Nvidia 4070 Ti Super | GPU | 31 | 91 | 89 | NA | 602 |
| Nvidia 5060 Ti | GPU | 17.4 | 46 | 50 | NA | 384 |
| Nvidia 5070 Ti | GPU | 32 | 98 | 98 | NA | 762 |
| Nvidia 4090 | GPU | 58 | 150 | 168 | NA | 912 |
| Nvidia 4090 (WSL2) | GPU | 53 |  |  |  | 885 |


NA = not available on the device. Usually shows up as error like these:

```
RuntimeError: "addmm_cuda" not implemented for 'Char'
RuntimeError: MPS device does not support mm for non-float inputs
```
