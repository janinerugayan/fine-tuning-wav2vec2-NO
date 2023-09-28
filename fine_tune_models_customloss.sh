cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossV7
cd fine-tuning_wav2vec2_customLossV7
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossV7
cd ../github/fine-tuning-wav2vec2-NO/

# CUDA_VISIBLE_DEVICES=1
python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossV7"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossV7/"\
                                           --num_train_epochs=10\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0.3\
                                           --use_asd_metric=1\
                                           --wandb_name="V7_customLoss_allDataSmall_300m_10ep"
                                        #    --export_log="./customLossV6.txt"


# cd ../../model_ckpts/
# mkdir fine-tuning_wav2vec2_origLossV5
# cd fine-tuning_wav2vec2_origLossV5
# mkdir runs
# cd ../../fine_tuned_models/
# mkdir wav2vec2_NO_origLossV5
# cd ../github/fine-tuning-wav2vec2-NO/

# python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                            --fine_tuned_model_ver="fine-tuning_wav2vec2_origLossV5"\
#                                            --export_model_dir="../../fine_tuned_models/wav2vec2_NO_origLossV5/"\
#                                            --num_train_epochs=10\
#                                            --learning_rate=1e-4\
#                                            --lambda_asd=0.3\
#                                            --use_asd_metric=0\
#                                            --wandb_name="V5_origLoss_allDataSmall_300m_10ep"




