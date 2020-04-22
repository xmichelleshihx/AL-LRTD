#!/bin/sh

python3 train_singlenet_phase_addnonlocal_AL.py \
    -l=5e-4 \
    -t=200 \
    -c=0 \
    -g=[0,1,2,3] \
    --adamstep=3 \
    -o=1\
    -m=1\
    -e=25 \
    --savedir="results_ResLSTM_Nolocal/round1" \
    --save_select_txt_path='nonlocalselect_txt/round1' \
    --json_name="17195_1572568659.835147.json" \
    --summary_dir='runs/'\
    --train_mode='RESLSTM_NOLOCAL'\
    --FT_checkpoint='results_ResLSTM_Nolocal/round1/RESLSTM/1572279640.0760705txtname8602_1571811265.416326.json_0.0005_tbs400_seq10_opt0_crop0_adjlr_sgdgamma0.1_sgdstep3_sgd_adjust_lr0_weight_decay0.0005/checkpoint_best-11.pt'\
    --sv_init_model='results_ResLSTM_Nolocal/round1/RESLSTM_NOLOCAL/1572599460.4324906txtname17195_1572568659.835147.json_0.0005_tbs400_seq10_opt1_crop0_adamgamma0.1_adamstep3_adamweightdecay0.0001_block_num1/checkpoint_best-23.pt'\
    -old=True \
    # --train_mode='RESLSTM_NOLOCAL_dropout0.2'\
    # --FT_checkpoint='results_ResLSTM_Nolocal/round1/RESLSTM/1572712641.20047txtname34381_1572702907.6751492.json_0.0005_tbs400_seq10_opt0_crop0_sgdstep3_sgd_gamma0.1_sgd_adjust_lr0_weight_decay0.0005/checkpoint_best-21.pt'\
    
    
    
    
    

