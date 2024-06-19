cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossTRIAL35_aulus5
cd fine-tuning_wav2vec2_customLossTRIAL35_aulus5
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossTRIAL35_aulus5
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossTRIAL35_aulus5"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossTRIAL35_aulus5/"\
                                           --num_train_epochs=3\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0.3\
                                           --use_asd_metric=1\
                                           --wandb_name="TRIAL35_customLoss_allDataSmall_300m_3ep_batched_aulus5"\
                                           --export_log="./loss_logs/customLossTRIAL35_aulus5.txt"


# for n in {1..5}
# do

#     echo FINE-TUNING MODEL CUSTOM LOSS w/ LAMBDA 0.$n

#     cd ../../model_ckpts/
#     mkdir fine-tuning_wav2vec2_customLossTRIAL21_titan1_lambda0p${n}
#     cd fine-tuning_wav2vec2_customLossTRIAL21_titan1_lambda0p${n}
#     mkdir runs
#     cd ../../fine_tuned_models/
#     mkdir wav2vec2_NO_customLossTRIAL21_titan1_lambda0p${n}
#     cd ../github/fine-tuning-wav2vec2-NO/

#     python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                             --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossTRIAL21_titan1_lambda0p${n}"\
#                                             --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossTRIAL21_titan1_lambda0p${n}/"\
#                                             --num_train_epochs=3\
#                                             --learning_rate=1e-4\
#                                             --lambda_asd=0.${n}\
#                                             --use_asd_metric=1\
#                                             --wandb_name="TRIAL21_lambda0p${n}_customLoss_allDataSmall_300m_3ep_batched_titan1"\
#                                             --export_log="./loss_logs/customLossTRIAL21_titan1_lambda0p${n}.txt"

# done






