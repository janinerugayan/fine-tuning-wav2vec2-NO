cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossTRIAL16_deepthought
cd fine-tuning_wav2vec2_customLossTRIAL16_deepthought
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossTRIAL16_deepthought
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossTRIAL16_deepthought"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossTRIAL16_deepthought/"\
                                           --num_train_epochs=1\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0.5\
                                           --use_asd_metric=1\
                                           --wandb_name="TRIAL16_customLoss_allDataSmall_300m_1ep_batched_deepthought"\
                                           --export_log="./loss_logs/TRIAL16_deepthought_customLoss.txt"

# cd ../../model_ckpts/
# mkdir fine-tuning_wav2vec2_origLossTRIAL5
# cd fine-tuning_wav2vec2_origLossTRIAL5
# mkdir run[srese
# cd ../../fine_tuned_models/
# mkdir wav2vec2_NO_origLossTRIAL5
# cd ../github/fine-tuning-wav2vec2-NO/

# python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                            --fine_tuned_model_ver="fine-tuning_wav2vec2_origLossTRIAL5"\
#                                            --export_model_dir="../../fine_tuned_models/wav2vec2_NO_origLossTRIAL5/"\
#                                            --num_train_epochs=1\
#                                            --learning_rate=1e-4\
#                                            --lambda_asd=0.1\
#                                            --use_asd_metric=1\
#                                            --wandb_name="TRIAL5_origLoss_allDataSmall_300m_3ep"\
#                                            --export_log="./loss_logs/TRIAL5_origLoss.txt"

# for n in {1..5}
# do

#     echo FINE-TUNING MODEL CUSTOM LOSS w/ LAMBDA 0.$n

#     cd ../../model_ckpts/
#     mkdir fine-tuning_wav2vec2_customLossV12_lambda0p${n}
#     cd fine-tuning_wav2vec2_customLossV12_lambda0p${n}
#     mkdir runs
#     cd ../../fine_tuned_models/
#     mkdir wav2vec2_NO_customLossV12_lambda0p${n}
#     cd ../github/fine-tuning-wav2vec2-NO/

#     python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                             --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossV12_lambda0p${n}"\
#                                             --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossV12_lambda0p${n}/"\
#                                             --num_train_epochs=3\
#                                             --learning_rate=1e-4\
#                                             --lambda_asd=0.${n}\
#                                             --use_asd_metric=1\
#                                             --wandb_name="V12_lambda0p${n}_customLoss_allDataSmall_300m_3ep"\
#                                             --export_log="./loss_logs/V12_lambda0p${n}_customLoss.txt"

# done






