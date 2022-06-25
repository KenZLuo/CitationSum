python src/train.py  -task abs -mode train -bert_data_path /data/xieqianqian/cisum/PubMed/transductive/  -dec_dropout 0.4 -model_path /data/xieqianqian/cisum/model/ -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 200 -batch_size 1200 -train_steps 200000 -report_every 50 -accum_count 50 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 640 -visible_gpus 1 -log_file /data/xieqianqian/cisum/pubmed.log #-train_from /data/xieqianqian/cisum/model/model_step_800.pt
