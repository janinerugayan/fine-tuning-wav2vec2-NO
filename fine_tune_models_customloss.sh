cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossTRIAL4
cd fine-tuning_wav2vec2_customLossTRIAL4
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossTRIAL4
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossTRIAL4"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossTRIAL4/"\
                                           --num_train_epochs=3\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0.1\
                                           --use_asd_metric=1\
                                           --wandb_name="TRIAL4_customLoss_allDataSmall_300m_3ep"\
                                           --export_log="./loss_logs/TRIAL4_customLoss.txt"

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






