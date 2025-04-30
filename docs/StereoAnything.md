<div align="center">
<h2>Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data</h2>

[**Xianda Guo**](https://scholar.google.com.hk/citations?hl=zh-CN&user=jPvOqgYAAAAJ)<sup>1,* </sup> · **Chenming Zhang**<sup>2,3,* </sup> · [**Youmin Zhang**](https://youmi-zym.github.io/)<sup>4,5</sup> · **Dujun Nie**<sup>6</sup> · **Ruilin Wang**<sup>6</sup>   
[**Wenzhao Zheng**](https://wzzheng.net/)<sup>7</sup> · [**Matteo Poggi**](https://mattpoggi.github.io/)<sup>4</sup> · [**Long Chen**](https://scholar.google.com.hk/citations?hl=zh-CN&user=jzvXnkcAAAAJ)<sup>6,2,3,&dagger;</sup>  

<sup>1</sup>Wuhan University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Xi'an Jiaotong University&emsp;&emsp;&emsp;&emsp;<sup>3</sup>Waytous&emsp;&emsp;&emsp;&emsp;<sup>4</sup>University of Bologna  
<sup>5</sup>Rock Universe&emsp;&emsp;&emsp;&emsp; <sup>6</sup> Institute of Automation, Chinese Academy of Sciences&emsp;&emsp;&emsp;&emsp; <sup>7</sup>University of California, Berkeley&emsp;&emsp;&emsp;&emsp;


<a href="https://arxiv.org/pdf/2411.14053"><img src='https://img.shields.io/badge/arXiv-Stereo Anything-red' alt='Paper PDF'></a> <a href='https://github.com/XiandaGuo/OpenStereo'><img src='https://img.shields.io/badge/Code-Stereo Anything-green' alt='Project Page'></a>
<a href='https://huggingface.co/papers/2411.14053'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow'></a> &nbsp;

</div>

This work presents Stereo Anything, a highly practical solution for stereo estimation by training on a combination of  **20M+ unlabeled images**.

![teaser](../misc/StereoAnything.png)

## News
* **2024-12-3:** Checkpoint of Stereo Anything is [here](https://drive.google.com/file/d/18BBk2y7f86PgiEBij3SlCSBwDIurp0K7/view?usp=sharing)
* **2024-11-26:** Code of [Stereo Anything](https://github.com/XiandaGuo/OpenStereo/cfgs/nerf) is released.
* **2024-11-14:** Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data, [*Paper*](https://arxiv.org/abs/2411.14053).

## Performance

Here we compare our Stereo Anything with the previous best model.

| Method               | K12   | K15   | Midd  | E3D   | DR    | Mean  |
|--------|-------|---------|-------|-------|-------|-------|
| PSMNet              | 30.51  | 32.15 | 33.53 | 18.02 | 36.19 | 30.08 |
| CFNet               | 13.64 | 12.09 | 23.91 |  7.67 | 27.26 | 16.91 |
| GwcNet              | 23.05 | 25.19 | 29.87 | 14.54 | 35.40 | 25.61 |
| COEX                | 12.08 | 11.01 | 25.17 | 11.43 | 24.17 | 16.77 |
| FADNet++            | 11.31 | 13.23 | 24.07 | 22.48 | 20.50 | 18.32 |
| Cascade             | 11.86 | 12.06 | 27.39 | 11.62 | 27.65 | 18.12 |
| LightStereo-L       |  6.41 |  6.40 | 17.51 | 11.33 | 21.74 | 12.68 |
| IGEV                |  4.88 |  5.16 |  8.47 |  3.53 |  **6.20** |  5.67 |
| StereoBase          |  4.85 |  5.35 |  9.76 |  3.12 | 11.84 |  6.98 |
| NMRFStereo          |  **4.20** |  5.10 |  7.50 |  3.80 | 11.92 |  6.50 |
| NMRFStereo*      |  8.67 |  7.46 | 16.36 | 23.46 | 34.58 | 18.11 |
| **StereoAnything**  |  4.29 |  **4.31** |  **6.96** |  **1.84** |  7.64 |  **5.01** |

We highlight the **best** results in **bold** (**better results**: $\downarrow$).

## Pre-trained models

We provide the models for robust stereo disparity estimation:


You can easily load our pre-trained models by:
```python
python tools/infer.py --cfg_file cfgs/nmrf/nmrf_swint_sceneflow.py \
--pretrained_model Your_model_path \
--left_img_path your/path \
--right_img_path your/path \
--savename your/path
```


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{guo2024stereo,
  title={Stereo Anything: Unifying Stereo Matching with Large-Scale Mixed Data},
  author={Guo, Xianda and Zhang, Chenming and Zhang, Youmin and Nie, Dujun and Wang, Ruilin and Zheng, Wenzhao and Poggi, Matteo and Chen, Long},
  journal={arXiv preprint arXiv:2411.14053},
  year={2024}
}
```
