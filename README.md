# Cross-lingual Cross-modal Pretraining for Multimodal Retrieval
This repository contains PaddlePaddle code that supports experiments in our NAACL 2021 paper: Cross-lingual Cross-modal Pretraining for Multimodal Retrieval. 
Note: Pytorch version is available upon request.

## Usage
Data and pretrained model paths are changable in configuraiton files in paddle_version/cfgs. 
### Pretraining Dataset:
We follow the steps [here](https://github.com/jackroos/VL-BERT/tree/master/data/conceptual-captions) to download and extract visual features.

* [English conceptual caption dataset](https://github.com/igorbrigadir/DownloadConceptualCaptions)
* [English SBU captions](http://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz)

### Finetune Dataset:
* [MS coco](https://cocodataset.org/#download), we use train2014, val2014 and test2015.
* [Multi30K](https://github.com/multi30k/dataset), we use train, val and test2016. 

The visual features are extracted in the same way as pretraining data.  

### Pretraining
```
bash scripts/pretrain.sh
```
### Fine-tuning
```
bash scripts/finetune.sh YOUR_PRETRAINED_MODEL_FILE YOUR_FINETUNE_CONFIG_FILE  YOUR_CHECKPOINT_DIR
```
where YOUR_FINETUNE_CONFIG_FILE is a configuration file (e.g., ./paddle_version/cfgs/xlretcoco/base_ret_ja_16x16G_fp32.yaml).

## Acknowledgements
Special thanks to the following Pytorch code that help us develop our work.
* [VL-BERT](https://github.com/jackroos/VL-BERT)
* [SCAN](https://github.com/kuanghuei/SCAN) 

## TODO
* Distributed version
* Scripts to automatically download, process and extract visual features. 

## Reference
If you find the work useful, please consider citing it as following:
```
@inproceedings{DBLP:conf/naacl/FeiYL21,
  author    = {Hongliang Fei and Tan Yu and Ping Li},
  title     = {Cross-lingual Cross-modal Pretraining for Multimodal Retrieval},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2021, Online, June 6-11, 2021},
  pages     = {3644--3650},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.naacl-main.285},
  doi       = {10.18653/v1/2021.naacl-main.285}
}
```
