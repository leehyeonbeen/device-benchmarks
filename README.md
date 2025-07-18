# Device Benchmarks

This repo is forked from: https://github.com/chsasank/device-benchmarks

I customized printings as below:

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



NA = not available on the device. Usually shows up as error like these:

```
RuntimeError: "addmm_cuda" not implemented for 'Char'
RuntimeError: MPS device does not support mm for non-float inputs
```
