# GraphGPT
Implementation of GraphGPT: Graph enhanced GPT for conditioned molecular generation

## generate:
One can specify different translate configurations in `test.sh` as the sample code below.

To replicate our results, download the pre-trained checkpoints from [here](https://drive.google.com/file/d/1FM-QtH2Bqy2VKT9eLgbgPvzsjWOWdmmR/view?usp=sharing).


```
python test.py \
  --batch_size_val 8 \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <your_intermediate_folder> \
  --checkpoint_dir <your_checkpoint_folder> \
  --checkpoint <your_target_checkpoint> \
  --known_class False \
  --beam_size 10 \
  --search_strategy False \
```

## Data:
Download the raw reaction dataset from [here](https://drive.google.com/drive/folders/1tpeOx2R_sUU0KhwnaLpyIy1iFtDifAGM?usp=sharing) or [Dai et al.](https://github.com/Hanjun-Dai/GLN) and put it into your data directory.

## Train:
One can specify different model and training configurations in `train.sh`. Below is a sample code that calls `train.py`. Simply run `./start.sh` for training.


```
python train.py \
  --encoder_num_layers 8 \
  --decoder_num_layers 8 \
  --heads 8 \
  --max_step 900000 \
  --batch_size_token 4096 \
  --save_per_step 5000 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --device cuda \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <your_data_folder> \
  --intermediate_dir <your_vocab_folder> \
  --checkpoint_dir <your_checkpoint_folder> \
  --checkpoint <your_previous_checkpoint> 
```



Node: 
Our code was developed with reference to the code written by [Wan et al.](https://proceedings.mlr.press/v162/wan22a.html), and we would like to express our gratitude to them.
