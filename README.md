DisenCDR
===

The source code is for the paper: “DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation” accepted in SIGIR 2022 by Jiangxia Cao, XiXun Lin, Xin Cong, Jing Ya, Tingwen Liu and Bin Wang.

```
@inproceedings{cao2022disencdr,
  title={DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation},
  author={Cao, Jiangxia and Lin, Xixun and Cong, Xin and Ya, Jing and Liu, Tingwen and Wang, Bin},
  booktitle={International Conference on Research on Development in Information Retrieval (SIGIR)},
  year={2022}
}
```

Requirements
---

Python=3.7.9

PyTorch=1.6.0

Scipy = 1.5.2

Numpy = 1.19.1

Usage
---

To run this project, please make sure that you have the following packages being downloaded. Our experiments are conducted on a PC with an Intel Xeon E5 2.1GHz CPU, 256 RAM and a Tesla V100 32GB GPU. 

Running example:

```shell
CUDA_VISIBLE_DEVICES=0 python -u train_rec.py --dataset sport_cloth > sport_cloth.log 2>&1&
```


