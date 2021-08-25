# VRC
**Official implementation of the Few-shot Visual Relationship Co-localization (ICCV 2021) paper**

[project page](https://vl2g.github.io/projects/vrc/) | [paper](https://vl2g.github.io/projects/vrc/docs/VRC-ICCV2021.pdf)

## Requirements
* Use **python >= 3.8.5**. Conda recommended : [https://docs.anaconda.com/anaconda/install/linux/](https://docs.anaconda.com/anaconda/install/linux/)

* Use **pytorch 1.7.0 CUDA 10.2**

* Other requirements from 'requirements.txt'

**To setup environment**
```
# create new env vrc
$ conda create -n vrc python=3.8.5

# activate vrc
$ conda activate vrc

# install pytorch, torchvision
$ conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch

# install other dependencies
$ pip install -r requirements.txt
```

## Training

### Preparing dataset
- Download VG images from [https://visualgenome.org/](https://visualgenome.org/)

- Extract faster_rcnn features of VG images using [data_preparation/vrc_extract_frcnn_feats.py](data_preparation/vrc_extract_frcnn_feats.py). Please follow instructions [here](data_preparation/README.md).

- Download VrR-VG dataset from [http://vrr-vg.com/](http://vrr-vg.com/) or [Google Drive Link](https://drive.google.com/file/d/1X7lYDviVKJI9bGmQAbQikTM271P3aoWZ/view?usp=sharing)

### Training VR Encoder (VTransE)

#### Training parameters
To check and update training, model and dataset parameters see [VR_Encoder/configs](VR_Encoder/configs)

#### To train VR Encoder: 
```
$ python train_vr_encoder.py
```

### Training VR Similarity Network (Relation Network)

#### Training parameters
To check and update training, testing, model and dataset parameters see [VR_SimilarityNetwork/configs](VR_SimilarityNetwork/configs)

#### To train VR Similarity Network: 
```
$ python SimilarityNetworkTrain.py
```

#### To train VR Similarity Network (w/ concat as VR Encoding): 
```
$ python ConcatplusSimilarityNetworkTrain.py
```

#### To evaluate (set eval setting in [test_config.yaml](VR_SimilarityNetwork/configs/test_config.yaml))
```
$ python FullModelTest.py
```

## Cite
If you find this code/paper  useful for your research, please consider citing.
```
@InProceedings{teotiaMMM2021,
  author    = "Teotia, Revant and Mishra, Vaibhav and Maheshwari, Mayank and Mishra, Anand",
  title     = "Few-shot Visual Relationship Co-Localization",
  booktitle = "ICCV",
  year      = "2021",
}
```

## Acknowledgements
This repo uses https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark and scripts from https://github.com/facebookresearch/mmf for Faster R-CNN feature extraction. 

Code provided by https://github.com/zawlin/cvpr17_vtranse and https://github.com/yangxuntu/vrd helped in implementing VR encoder.


### Contact
For any clarification, comment, or suggestion please create an issue or contact [Revant](https://revantteotia.github.io/), [Vaibhav](https://www.linkedin.com/in/vaibhav-mishra-iitj/) or [Mayank](https://www.linkedin.com/in/maheshwarimayank333/).