# Using Diffusion Models to recover voice from 1/3 oct recordings

## Setup

The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```

## Paper Results Replication

### Datasets download

Download the LJSpeech dataset (2.6G) from:
https://keithito.com/LJ-Speech-Dataset/

The librispeech dataset "train-clean-100.tar.gz" (6.3G) and "test-clean.tar.gz" (346M) from:
https://www.openslr.org/12

The TAU Urban Acoustic Scenes 2020 Mobile, Development dataset (30.5GB) from:
https://dcase.community/challenge2020/task-acoustic-scene-classification

Put the two datasets in the "datasets" folder

### Experiment: training models

To create the Mel and Pinv-Mel dataset for each of the audio datasets, launch:

```
python3 exp_train_diffusion/dataset_tau.py
python3 exp_train_diffusion/dataset_ljspeech.py
python3 exp_train_diffusion/dataset_librispeech.py
```

You can eventually add the option `--audio_dataset PATH_TO_DATASET` to each of those commands to modify the default path.

To evaluate the baseline models (oracle and pinv), launch the following code:

```
python3 exp_train_diffusion/main_doce_diffusion_training.py -s baseline/tho_type=fast -c
```

To train the diffusion models launch the following code:

```
python3 exp_train_diffusion/main_doce_diffusion_training.py -s traindiff/tho_type=fast+learning_rate=-4+epoch=70+schedule=DDPM+diff_steps=1000+dataset=ljspeech -c
python3 exp_train_diffusion/main_doce_diffusion_training.py -s traindiff/tho_type=fast+learning_rate=-4+epoch=40+schedule=DDPM+diff_steps=1000+dataset=tau -c
python3 exp_train_diffusion/main_doce_diffusion_training.py -s traindiff/tho_type=fast+learning_rate=-4+epoch=40+schedule=DDPM+diff_steps=1000+dataset=librispeech -c

```

To create mel on eval datasets using the diffusion models, vocode the mels into audio using griffin lim and gomin, and calculate the metrics launch the following code:

```
python3 exp_train_diffusion/plot_result_tables.py -s --evaldiff/tho_type=fast+learning_rate=-4+epoch=70+schedule=DDPM+diff_steps=1000+dataset=ljspeech -c
python3 exp_train_diffusion/plot_result_tables.py -s evaldiff/tho_type=fast+learning_rate=-4+epoch=40+schedule=DDPM+diff_steps=1000+dataset=tau -c
python3 exp_train_diffusion/main_doce_diffusion_training.py -s evaldiff/tho_type=fast+learning_rate=-4+epoch=40+schedule=DDPM+diff_steps=1000+dataset=librispeech -c
```

### Experiment: results display

To display the metrics for each dataset, launch:

```
python3 exp_train_diffusion/plot_result_tables.py -s --evaldiff ljspeech
python3 exp_train_diffusion/plot_result_tables.py -s --evaldiff librispeech
```

To display the results for the preliminary perceptual evaluation (with 3 participants), launch:

```
perceptual_test_preliminary_reordering.py
perceptual_test_preliminary_wer_after_correction.py
perceptual_test_preliminary_statistics.py
```

To display the results from the main perceptual evaluation (with 20 participants), launch:

```
perceptual_test_reordering.py
perceptual_test_wer_after_correction.py
perceptual_test_statistics.py
```

### Experiment: figures

To reproduce the paper's figures in the "figures" folder, launch:
```
plot_spectro_icassp.py
plot_spectro_raw_icassp.py
```
