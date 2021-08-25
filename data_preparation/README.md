# Instructions to extract faster r-cnn features:
Check the script [vrc_extract_frcnn_feats.py](vrc_extract_frcnn_feats.py) and follow these instructions (also written as comments in the script)

0. Activate vrc conda environment
```
$ conda activate vrc
```

1. Install maskrcnn-benchmark : FRCNN Model
```	
$ git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git
$ cd vqa-maskrcnn-benchmark
$ python setup.py build
$ python setup.py develop
```
2. download pre-trained detectron weights
```
$ mkdir detectron_weights
$ wget -O detectron_weights/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth
$ wget -O detectron_weights/detectron_model.yaml  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
```

NOTE: just modify the code in /content/vqa-maskrcnn-benchmark/maskrcnn_benchmark/utils/imports.py, change PY3 to PY37

to run the script
```
$ python vrc_extract_frcnn_feats.py --image_dir=<path to images directory>
```
