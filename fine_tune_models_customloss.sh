# cd ../../model_ckpts/
# mkdir fine-tuning_wav2vec2_asd_trial71_origLoss_3ep_allDataSmall_aulus7
# cd fine-tuning_wav2vec2_asd_trial71_origLoss_3ep_allDataSmall_aulus7
# mkdir runs
# cd ../../fine_tuned_models/
# mkdir wav2vec2_NO_asd_trial71_origLoss_3ep_allDataSmall_aulus7
# cd ../github/fine-tuning-wav2vec2-NO/

# python3 fine_tuning_wav2vec2_customlossV5.py --original_model="facebook/wav2vec2-large-xlsr-53"\
#                                            --fine_tuned_model_ver="fine-tuning_wav2vec2_asd_trial71_origLoss_3ep_allDataSmall_aulus7"\
#                                            --export_model_dir="../../fine_tuned_models/wav2vec2_NO_asd_trial71_origLoss_3ep_allDataSmall_aulus7/"\
#                                            --num_train_epochs=3\
#                                            --learning_rate=1e-4\
#                                            --lambda_asd=0.7\
#                                            --use_asd_metric=0\
#                                            --wandb_name="asd_trial71_origLoss_3ep_allDataSmall_300m_batched_aulus7"\
#                                            --export_log="./loss_logs/asd_trial71_origLoss_3ep_allDataSmall_aulus7.txt"


cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_asd_trial76_masd_6ep_allDataSmall_aulus6
cd fine-tuning_wav2vec2_asd_trial76_masd_6ep_allDataSmall_aulus6
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_asd_trial76_masd_6ep_allDataSmall_aulus6
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_asd_trial76_masd_6ep_allDataSmall_aulus6"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_asd_trial76_masd_6ep_allDataSmall_aulus6/"\
                                           --num_train_epochs=6\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0\
                                           --use_asd_metric=1\
                                           --wandb_name="asd_trial76_masd_6ep_allDataSmall_300m_batched_aulus6"\
                                           --export_log="./loss_logs/asd_trial76_masd_6ep_allDataSmall_aulus6.txt"