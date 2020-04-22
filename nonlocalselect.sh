#!/bin/sh

python3 train_singlenet_phase_addnonlocal_AL.py \
    -l=5e-4 \
    --adamweightdecay=1e-4 \
    -t=128 \
    -c=0 \
    -g=[0,1,2,3] \
    --adamstep=3 \
    -e=25 \
    --savedir="results_ResLSTM_Nolocal/round2" \
    --save_select_txt_path='nonlocalselect_txt/round2' \
    --json_name="8594_1580362454.4802742.json" \
    --summary_dir='runs/'\
    --train_mode='RESLSTM_NOLOCAL'\
    --val_model_path='results_ResLSTM_Nolocal/round2/RESLSTM_NOLOCAL/1580388013.643609txtname8594_1580362454.4802742.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-8.pt'\
    --select_chose='non_local' \
    --is_save_json=False \

    
    
    
    

