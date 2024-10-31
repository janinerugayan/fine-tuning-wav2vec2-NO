# fine-tuned model path should be WITHOUT "/" at the end of the path in order to save transcriptions in unique csv files

cd logs
mkdir trial32_masd_RundkastOnly_aulus7
echo trial32_masd_RundkastOnly_aulus7
cd ../

python test_wav2vec2_scoreperutt_OneData.py --original_model="NbAiLab/nb-wav2vec2-320m-bokmaal"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_asd_trial32_masd_RundkastOnly_aulus7"\
                                    --log_file_name="test_log_wav2vec2_NO_trial32_masd_RundkastOnly_aulus7"\
                                    --log_dir="./logs/trial32_masd_RundkastOnly_aulus7/"\
                                    --get_orig_model_results=0