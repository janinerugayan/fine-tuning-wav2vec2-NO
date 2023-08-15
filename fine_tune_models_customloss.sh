# cd ../../model_ckpts/
# mkdir fine-tuning_wav2vec2_origLossV2
# cd fine-tuning_wav2vec2_origLossV2
# mkdir runs
# cd ../../fine_tuned_models/
# mkdir wav2vec2_NO_origLossV2
# cd ../github/fine-tuning-wav2vec2-NO/

# python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                            --fine_tuned_model_ver="fine-tuning_wav2vec2_origLossV2"\
#                                            --export_model_dir="../../fine_tuned_models/wav2vec2_NO_origLossV2/"\
#                                            --num_train_epochs=10\
#                                            --learning_rate=1e-4\
#                                            --use_asd_metric=0\
#                                            --wandb_name="V2_origLoss_allDataSmall_300m_10ep"



cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossV2
cd fine-tuning_wav2vec2_customLossV2
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossV2
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossV2"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossV2/"\
                                           --num_train_epochs=3\
                                           --learning_rate=1e-4\
                                           --use_asd_metric=1\
                                           --wandb_name="V2_customLoss_allDataSmall_300m_3ep"
