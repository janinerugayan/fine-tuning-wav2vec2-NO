# cd ../../model_ckpts/
# mkdir fine-tuning_wav2vec2_origLossV9
# cd fine-tuning_wav2vec2_origLossV9
# mkdir runs
# cd ../../fine_tuned_models/
# mkdir wav2vec2_NO_origLossV9
# cd ../github/fine-tuning-wav2vec2-NO/

# python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                            --fine_tuned_model_ver="fine-tuning_wav2vec2_origLossV9"\
#                                            --export_model_dir="../../fine_tuned_models/wav2vec2_NO_origLossV9/"\
#                                            --num_train_epochs=3\
#                                            --learning_rate=1e-4\
#                                            --lambda_asd=0.1\
#                                            --use_asd_metric=0\
#                                            --wandb_name="V9_origLoss_allDataSmall_300m_3ep"\
#                                            --export_log="./loss_logs/V9_origLoss.txt"

cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossV13
cd fine-tuning_wav2vec2_customLossV13
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossV13
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossV13"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossV13/"\
                                           --num_train_epochs=10\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0.1\
                                           --use_asd_metric=1\
                                           --wandb_name="V13_customLoss_allDataSmall_300m_10ep"\
                                           --export_log="./loss_logs/V13_customLoss.txt"

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






