# NCSNv2-TF2.0
Implementation of Noise Conditioned Score Network v2 in Tensorflow 2.0. This version is heavily borrowed from the reproducibility challenge [implementation](https://github.com/Xemnas0/NCSN-TF2.0) of NCSNv1.

[Original NCSNv1 Paper](https://arxiv.org/abs/1907.05600) | [Reprodcibility Challenge Paper](https://openreview.net/forum?id=SkxCSTqG6H) | [NCSNv2 Paper](https://arxiv.org/abs/2006.09011)

------------------------------------

```bash
usage: main.py [-h] [--experiment EXPERIMENT] [--dataset DATASET] [--model MODEL] [--filters FILTERS] [--num_L NUM_L] [--sigma_low SIGMA_LOW] [--sigma_high SIGMA_HIGH]
               [--sigma_sequence SIGMA_SEQUENCE] [--steps STEPS] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--samples_dir SAMPLES_DIR]
               [--checkpoint_dir CHECKPOINT_DIR] [--checkpoint_freq CHECKPOINT_FREQ] [--resume] [--resume_from RESUME_FROM] [--init_samples INIT_SAMPLES] [--k K]
               [--eval_setting EVAL_SETTING] [--ocnn] [--y_cond] [--max_to_keep MAX_TO_KEEP] [--split SPLIT] [--T T] [--eps EPS]

CLI Options

optional arguments:
  -h, --help            show this help message and exit
  --experiment EXPERIMENT
                        what experiment to run (default: train)
  --dataset DATASET     tfds name of dataset (default: 'mnist')
  --model MODEL         Model to use. Can be 'refinenet', 'baseline' (default: refinenet)
  --filters FILTERS     number of filters in the model. (default: 128)
  --num_L NUM_L         number of levels of noise to use (default: 10)
  --sigma_low SIGMA_LOW
                        lowest value for noise (default: 0.01)
  --sigma_high SIGMA_HIGH
                        highest value for noise (default: 1.0)
  --sigma_sequence SIGMA_SEQUENCE
                        can be 'geometric' or 'linear' (default: geometric)
  --steps STEPS         number of steps to train the model for (default: 200000)
  --learning_rate LEARNING_RATE
                        learning rate for the optimizer
  --batch_size BATCH_SIZE
                        batch size (default: 128)
  --samples_dir SAMPLES_DIR
                        folder for saving samples (default: ./samples/)
  --checkpoint_dir CHECKPOINT_DIR
                        folder for saving model checkpoints (default: ./saved_models/)
  --checkpoint_freq CHECKPOINT_FREQ
                        how often to save a model checkpoint (default: 5000 iterations)
  --resume              whether to resume from latest checkpoint (default: True)
  --resume_from RESUME_FROM
                        Step of checkpoint where to resume the model from. (default: latest one)
  --init_samples INIT_SAMPLES
                        Folder with images to be used as x0 for sampling with annealed langevin dynamics
  --k K                 number of nearest neighbours to find from data (default: 10)
  --eval_setting EVAL_SETTING
                        can be 'sample' or 'fid' (default: sample)
  --ocnn                whether to attach an ocnn to the model (default: False)
  --y_cond              whether the model is conditioned on auxiallary y information (default: False)
  --max_to_keep MAX_TO_KEEP
                        Number of checkopints to keep saved (default: 2)
  --split SPLIT         optional train/validation split percentages e.g. 0.9*train, 0.1*train (default: all train, no val set)
  --T T                 number of iterations to sample for each sigma (default: 100)
  --eps EPS             epsilon for generating samples (default: 2*1e-5)

  ```