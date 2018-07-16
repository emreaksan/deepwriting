# DeepWriting
Code for `DeepWriting: Making Digital Ink Editable via Deep Generative Modeling` [paper](https://arxiv.org/abs/1801.08379).

[![Watch the video](https://img.youtube.com/vi/NVF-1csvVvc/0.jpg)](https://www.youtube.com/watch?v=NVF-1csvVvc)

Implementation of conditional variational RNNs (C-VRNN).

## Dataset
We collected data from 94 authors by using [IAMOnDB](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) corpus. After discarding noisy samples of IAMOnDB, we compiled a dataset of 294 authors, fully segmented. You can download our preprocessed data from [project page](https://ait.ethz.ch/projects/2018/deepwriting/downloads/deepwriting_dataset.tar.gz).

The dataset is in compressed .npz format. Strokes, labels and statistics can be accessed by using <key, value> pairs. We split handwriting samples by end-of-characters such that each sample consists of ~300 strokes. Samples are then moved to the origin and represented by using the offset values between consecutive strokes.

For example, you can visualize validation samples with indices 1, 5 and 20 by running
```
python visualize_hw.py -D ./data/deepwriting_validation.npz -O ./data_images -S 1 5 20
```

If you use our data, we kindly ask you to cite our work, and also fill [IAMOnDB's registration form](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php) and follow their citation requirements since our dataset extends IAMOnDB.

## Training Model
1. Training details and hyper-parameters are defined in `config.py`.
2. Download training dataset and copy into `data` folder. Otherwise, don't forget to update `training_data` and `validation_data` entries in `config.py`.
3. Set `PYTHONPATH` to include `source`.
```
export PYTHONPATH=$PYTHONPATH:./source
```
4. Run training
```
python tf_train_hw.py -S <path_to_save_experiment>
```
5. If you want to continue training 
```
python tf_train_hw.py -S <path_to_save_experiment> -M <model_folder_name>
```

## Evaluating Model
1. You can either train a model or download our pretrained model from [project page](https://ait.ethz.ch/projects/2018/deepwriting/downloads/tf-1514981744-deepwriting_synthesis_model.tar.gz).
2. You can run
```
python tf_evaluate_hw.py -S <path_to_save_experiment> -M <model_folder_name> -QL
```
3. Evaluation options are defined in `tf_evaluate_hw.py` globally.


## Dependencies
1. Numpy
2. Tensorflow 1.2+ (not sure if earlier versions work.)
3. Matplotlib
4. OpenCV (pip install opencv-python is enough)
5. svgwrite

## Missing
We are not planning to release demo interface.

## Citation
If you use this code or dataset in your research, please cite us as follows:
```
@inproceedings{Aksan:2018:DeepWriting,
	author = {Aksan, Emre and Pece, Fabrizio and Hilliges, Otmar},
	title = {{DeepWriting: Making Digital Ink Editable via Deep Generative Modeling}},
	booktitle = {SIGCHI Conference on Human Factors in Computing Systems},
	series = {CHI '18},
	year = {2018},
	location = {Montr{\'}eal, Canada},
	publisher = {ACM},
	address = {New York, NY, USA},
}
```
