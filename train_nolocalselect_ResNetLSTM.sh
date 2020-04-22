#!/bin/sh

python3 train_singlenet_phase_addnonlocal_AL.py \
    -l=5e-4 \
    -t=400 \
    -c=0 \
    -g=[0,1,2,3] \
    --adamstep=0 \
    --sgdadjust=0 \
    --sgdstep=3 \
    --weightdecay=5e-4 \
    -o=0\
    -m=1\
    -e=25 \
    --savedir="results_ResLSTM_Nolocal/round2" \
    --save_select_txt_path='nonlocalselect_txt/round2' \
    --json_name="42966_1580647357.8568866.json" \
    --summary_dir='runs/'\
    --train_mode='RESLSTM'\
    # -old=True \
    
    
    
    

