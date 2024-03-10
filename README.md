# ISDM
Imaging through scattering media via generative diffusion model

**Paper**: Imaging through scattering media via generative diffusion model

**Authors**: Z. Chen, B. Lin, S. Gao, W. Wan*, Q. Liu*   

Applied Physics Letters (Vol.124, Issue 5)     
https://pubs.aip.org/aip/apl/article/124/5/051101/3176612/Imaging-through-scattering-media-via-generative        

Date : Jan-29-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, Department of Electronic Information Engineering, Nanchang University.  

# Checkpoints
ISDM : We provide pretrained checkpoints. You can download pretrained models from [Baidu cloud] (https://pan.baidu.com/s/1CX7xCh1uJl-h5ZojO5SkNQ?pwd=hdtp) Extract the code (hdtp)

# Dataset
The dataset used to train the model in this experiment is MNIST dataset.

place the dataset in the train file under the church folder.

# Train:
python main.py --config=configs/ve/church_ncsnpp_continuous.py --workdir=exp_train_church_max1_N1000 --mode=train --eval_folder=result

# Test:
python ISDM_reconstruction.py

## Structure diagram of imaging through scattering media
<div align="center"><img src="https://github.com/yqx7150/ISDM/blob/main/ISDM/Figures/FIG1.png"> </div>
    
## Training and reconstruction flow chart of ISDM
<div align="center"><img src="https://github.com/yqx7150/ISDM/blob/main/ISDM/Figures/FIG2.png"> </div>

## The scattering imaging system
<div align="center"><img src="https://github.com/yqx7150/ISDM/blob/main/ISDM/Figures/FIG6.png"> </div>

## Recovery results of measured scatter data by different methods
<div align="center"><img src="https://github.com/yqx7150/ISDM/blob/main/ISDM/Figures/FIG7.png"> </div>

(a) USAF resolution targets, (b) speckles, (c) LR, (d) HIO, (e) TC, and (f) ISDM method. From top to bottom, G1E2: element 2 of group 1, G1E6: element 6 of group 1, G2E5: element 5 of group 2, and G3E2: element 2 of group 3.


