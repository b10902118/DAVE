# Detectron2 Installation
1. change `>=3.9` in detectron2/setup.py to `>=3.8`
2. prepare `nvcc 11.8`, `gcc-11` (a whole package). Install with pip. Ensure the install script use the prepared compilers (`build.ninja` file).

# DAVE Installation Guide

To install and set up the DAVE environment, follow these steps:

1. **Create a Conda environment and install dependencies:**

    ```bash
    conda create -n dave python==3.8
    conda activate dave
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install numpy
    conda install scikit-image
    conda install scikit-learn
    conda install tqdm
    conda install pycocotools
    ```
   Additionally, if you will run text-prompt-based counting install also:
   ```bash
   conda install transformers
   ```

2. **Download the models:**

    Download the pre-trained models from [here](https://drive.google.com/drive/folders/10O4SB3Y380hcKPIK8Dt8biniVbdQ4dH4?usp=sharing) and configure dataset and model path in the `utils/argparser.py`.

3. **Run the scripts:**

   You can run the provided scripts for zero-, one-, few-shot counting or text prompt based counting. Additionally, you can use the `demo.py` file to run the model on custom images.

4. **Evaluation**
   To evaluate the results install `detectron2`, and run the script `/utils/eval.py`

## Demo
   After succesfully completing step 1 and 2 of the installation, you can run `demo.py` on your images or provided examples:

   ```bash
   python demo.py --skip_train --model_name DAVE_3_shot --model_path material --backbone resnet50 --swav_backbone --reduction 8 --num_enc_layers 3 --num_dec_layers 3 --kernel_dim 3 --emb_dim 256 --num_objects 3 --num_workers 8 --use_query_pos_emb --use_objectness --use_appearance --batch_size 1 --pre_norm
   ```
#### Zero Shot Demo
    ```
    python demo_zero.py --img_path <input-file> --show --zero_shot --two_passes --skip_train --model_name DAVE_0_shot --model_path material --backbone resnet50 --swav_backbone --reduction 8 --num_enc_layers 3 --num_dec_layers 3 --kernel_dim 3 --emb_dim 256 --num_objects 3 --num_workers 8 --use_objectness --use_appearance --batch_size 1 --pre_norm
    ```

#### Test new model
    ```
    $ CUDA_VISIBLE_DEVICES=3 python demo_zero.py --img_path ./material/741.jpg  --zero_shot --two_passes --skip_train --model_name DAVE_0_shot --model_path /tmp2/r13922043/DAVE/material  --backbone resnet50 --swav_backbone --reduction 8 --num_enc_layers 3 --num_dec_layers 3 --kernel_dim 3 --emb_dim 256 --num_objects 3 --num_workers 8 --use_objectness --use_appearance --batch_size 1 --pre_norm --prompt_shot

    $ CUDA_VISIBLE_DEVICES=3 python demo.py --skip_train --model_name DAVE_3_shot --model_path /tmp2/r13922043/DAVE/material --backbone resnet50 --swav_backbone --reduction 8 --num_enc_layers 3 --num_dec_layers 3 --kernel_dim 3 --emb_dim 256 --num_objects 3 --num_workers 8 --use_query_pos_emb --use_objectness --use_appearance --batch_size 1 --pre_norm
    ```

## Citation

If you use DAVE in your research, please cite the following paper:

```bibtex
@InProceedings{Pelhan_2024_CVPR,
author = {Jer Pelhan and Alan Lukežič and Vitjan Zavrtanik and Matej Kristan},
title = {DAVE – A Detect-and-Verify Paradigm for Low-Shot Counting},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2024}
}
```
