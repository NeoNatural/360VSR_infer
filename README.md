# Spherical Signal Super-resolution with Proportioned Optimisation (S3PO)

Code Repository for S3PO video super-resolution model for panaromic videos accepted to be published in **IEEE Transactions on Multimedia** entitled **"Omnidirectional Video Super-resolution using Deep Learning"**.

Please refer to the paper for details - https://www.techrxiv.org/articles/preprint/Omnidirectional_Video_Super-Resolution_using_Deep_Learning/20494851

To use this work or dataset please cite our paper:

@ARTICLE{10102571,
  author={Baniya, Arbind Agrahari and Lee, Tsz-Kwan and Eklund, Peter W. and Aryal, Sunil},
  journal={IEEE Transactions on Multimedia}, 
  title={Omnidirectional Video Super-Resolution using Deep Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TMM.2023.3267294}}

@article{Agrahari Baniya2023,
author = "Arbind Agrahari Baniya and Glory Lee and Peter Eklund and Sunil Aryal",
title = "{Omnidirectional Video Super-Resolution using Deep Learning}",
year = "2023",
month = "4",
url = "https://www.techrxiv.org/articles/preprint/Omnidirectional_Video_Super-Resolution_using_Deep_Learning/20494851",
doi = "10.36227/techrxiv.20494851.v2"
}

## 推理（视频输入）

使用提供的 `inference.py` 可以直接对全景 MP4 视频进行超分辨率推理。示例命令：

```bash
python inference.py \
  --input-video /path/to/input_panorama.mp4 \
  --model-path /path/to/checkpoint.pth \
  --scale 4 \
  --fps 25 \
  --device cuda
```

可选参数 `--output-path` 用于指定输出视频完整路径和文件名；若不提供，则输出会与输入视频同目录保存，并在文件名后追加 `_sr` 后缀以示区分。
