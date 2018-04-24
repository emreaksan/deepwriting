# DeepWriting
Code for `DeepWriting: Making Digital Ink Editable via Deep Generative Modeling` [paper](https://arxiv.org/abs/1801.08379).

[![Watch the video](https://img.youtube.com/vi/NVF-1csvVvc/0.jpg)](https://www.youtube.com/watch?v=NVF-1csvVvc)

Implementation of conditional variational RNNs (C-VRNN).

## Dataset
We collected data from 94 authors by using [IAMOnDB](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) corpus. After discarding noisy samples of IAMOnDB, we compiled a dataset of 294 authors, fully segmented. For now, we release a very small subset of preprocessed data (i.e., test split). Full dataset in raw format will be shared when we resolve permission issues.

## Pretrained Model
1. You can download a pretrained model from [our project page](https://ait.ethz.ch/projects/2018/deepwriting/downloads/tf-1514981744-deepwriting_synthesis_model.tar.gz).
2. Either move under `<repository_path>/runs/` or update `validation data path.` in config.json. 
3. You can run
```
python tf_evaluate_hw.py -S <path_to_model_folder> -M tf-1514981744-deepwriting_synthesis_model -QL
```


## Dependencies
1. Numpy
2. Tensorflow 1.2+ (not sure if earlier versions work.)
3. Matplotlib
4. OpenCV (pip install opencv-python is enough)
5. svgwrite
