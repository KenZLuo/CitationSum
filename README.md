# contrast_gcn_sum

**This code is for the implementation of the contrastive variational graph auto-encoder for scientific paper summarization with citation graph**

Some codes are from PreSum:https://github.com/nlpyang/PreSumm and CGSum: https://github.com/ChenxinAn-fdu/CGSum

**Python version**: This code is in Python3.6

**Package Requirements**: torch==1.1.0 pytorch_transformers tensorboardX multiprocess pyrouge

## Data Preparation For SSN
### Option 1: download the processed data

[Pre-processed data](https://drive.google.com/open?id=1DN7ClZCCXsk2KegmC6t4ClBwtAf5galI)

unzip the zipfile and put all `.pt` files into `bert_data`

### Option 2: process the data yourself

#### Step 1 Download Stories
Download and unzip the SSN (including inductive and transductive) from [here](https://github.com/ChenxinAn-fdu/CGSum).

####  Step 2. Format to PyTorch Files
```
python preprocess.py -mode format_cite -raw_path RAW_PATH -save_path BERT_DATA_PATH  -lower -n_cpus 8 -log_file ../logs/preprocess.log
```

* `RAW_PATH` is the directory containing raw files (`../inductive`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

## Model Training

### Abstractive Setting

```
python train.py  -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 120 -train_steps 200000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0,1,2,3  -log_file ../ssn.log
```

## Model Evaluation
### SSN
```
 python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_ssn -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 130 -result_path ../logs/
```
* `-mode` can be {`validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use
* `MODEL_PATH` is the directory of saved checkpoints
* use `-mode valiadte` with `-test_all`, the system will load all saved checkpoints and select the top ones to generate summaries (this will take a while)

