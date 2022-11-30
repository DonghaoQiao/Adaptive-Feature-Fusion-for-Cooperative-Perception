# Adaptive Feature Fusion for Cooperative Perception using LiDAR Point Clouds [[WACV2023](https://wacv2023.thecvf.com)][[paper](https://arxiv.org/abs/2208.00116)]

Please refer to the [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) repository for setup and installation.

## Qualitative Results
<p align="center">
    <img src="/images/cood1_1.png" width="45%" alt="">
    <img src="/images/cood1_7.png" width="45%" alt="">
    <img src="/images/cood2_1.png" width="45%" alt="">
    <img src="/images/cood2_7.png" width="45%" alt="">
    <img src="/images/cood3_1.png" width="45%" alt="">
    <img src="/images/cood3_7.png" width="45%" alt="">
    <img src="/images/cood4_1.png" width="45%" alt="">
    <img src="/images/cood4_7.png" width="45%" alt="">
    <div align="center">Single Vehicle Perception v.s. Cooperative Perception</div>
</p>


## Training
```python
python3 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_spatialcooper.yaml
```

## Evaluation
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder.
- Testing dataset path: `opv2v_data_dumping/test`.
- Culver City dataset path: `'opv2v_data_dumping/test_culver_cit`.

```python
python3 opencood/tools/inference.py --model_dir opencood/logs/point_pillar_spatialcooper/ --fusion_method intermediate
```

## Acknowledgement
- Xu, R., Xiang, H., Xia, X., Han, X., Li, J. and Ma, J., 2022, May. Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 2583-2589). IEEE.