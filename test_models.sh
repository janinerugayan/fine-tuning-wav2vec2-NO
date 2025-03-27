# fine-tuned model path should be WITHOUT "/" at the end of the path in order to save transcriptions in unique csv files

# cd logs
# mkdir trial75_mwer_6ep_allDataSmall_titan1
# echo trial75_mwer_6ep_allDataSmall_titan1
# cd ../

# python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaanl"\
#                                     --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_asd_trial75_mwer_6ep_allDataSmall_titan1"\
#                                     --log_file_name="test_log_wav2vec2_NO_trial75_mwer_6ep_allDataSmall_titan1"\
#                                     --log_dir="./logs/trial75_mwer_6ep_allDataSmall_titan1/"\
#                                     --get_orig_model_results=0

cd logs
mkdir trial68_DEVSET_6ep_allDataSmall_aulus6
echo trial68_DEVSET_6ep_allDataSmall_aulus6
cd ../

python test_wav2vec2_devset.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaanl"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_asd_trial68_masd_6ep_allDataSmall_aulus6"\
                                    --log_file_name="test_log_wav2vec2_NO_trial68_DEVSET_6ep_allDataSmall_aulus6"\
                                    --log_dir="./logs/trial68_DEVSET_6ep_allDataSmall_aulus6/"\
                                    --get_orig_model_results=0\
                                    --dev_set_path="../../datasets/IS25_devset/trial68_dev_set.csv"

