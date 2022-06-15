python src/train.py  -task abs -mode train -bert_data_path /data/xieqianqian/cisum/transductive/ -dec_dropout 0.4 -model_path /data/xieqianqian/cisum/models/ -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 2000 -batch_size 400 -train_steps 200000 -report_every 50 -accum_count 10 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 614 -visible_gpus 0,1,2,3,5,6 -generator_shard_size 64 -log_file /data/xieqianqian/cisum/ssn.log #-train_from /data/xieqianqian/cisum/models/model_step_112000.pt
