# ISDM

**Paper**: Imaging through scattering media via generative diffusion model

**Authors**: Z. Chen, B. Lin, S. Gao, W. Wan*, Q. Liu*   

Applied Physics Letters (Vol.124, Issue 5)     
https://pubs.aip.org/aip/apl/article/124/5/051101/3176612/Imaging-through-scattering-media-via-generative        

Date : Jan-29-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, Department of Electronic Information Engineering, Nanchang University.  

The scattering medium scrambles the light paths emitted from the targets into speckle patterns, leading to a signifi-cant degradation of the target image. Conventional iterative phase recovery algorithms typically yield low-quality reconstructions. On the other hand, supervised learning methods exhibit limited generalization capabilities in the con-text of image reconstruction. An approach is proposed for achieving high-quality reconstructed target images through scattering media using a diffusion generative model. The gradient distribution prior information of the target image is modeled using a scoring function, which is then utilized to constrain the iterative reconstruction process. The high-quality target image is generated by alternatively performing the stochastic differential equation solver and physical model-based data consistency steps. Simulation and experimental validation demonstrate that the proposed method achieves better image reconstruction quality compared to traditional methods, while ensuring generalization capabili-ties.  

# Structure diagram of imaging through scattering media. 
![FIG1](https://github.com/zhaoyun1201/ISDM/assets/106502358/8b578514-a8fd-414f-9638-d0d5c6481c05)

# Network structure of ISDM.
![FIG2](https://github.com/zhaoyun1201/ISDM/assets/106502358/ae7e27d1-3077-4954-ab56-c1c2ad96aa96)

# The scattering imaging system.
![FIG6](https://github.com/zhaoyun1201/ISDM/assets/106502358/1194badf-f521-47f1-b9ea-a1146a425e28)

# Recovery results of measured scatter data by different methods.
![FIG7](https://github.com/zhaoyun1201/ISDM/assets/106502358/328c5f4b-eb00-4360-8f50-2861b09f2d38)

# Requirements and Dependencies
python==3.7.11<br>
Pytorch==1.7.0<br>
tensorflow==2.4.0<br>
torchvision==0.8.0<br>
tensorboard==2.7.0<br>
scipy==1.7.3<br>
numpy==1.19.5<br>
ninja==1.10.2<br>
matplotlib==3.5.1<br>
jax==0.2.26<br>

# Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu cloud] 
(https://pan.baidu.com/s/1YBI7PLmGyzId-YLg_BW1sA) Extract the code (im5k)

# Dataset

The dataset used to train the model in this experiment is MNIST.

place the dataset in the train file under the Train_data folder.

# Train:
python main.py --config=configs/ve/church_ncsnpp_continuous.py --workdir=exp_train_MNIST_max1_N1000 --mode=train --eval_folder=result

# Test:
Use the following command to test： python A_PCsampling.py

# Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.







