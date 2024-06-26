# fine-tuned model path should be WITHOUT "/" at the end of the path in order to save transcriptions in unique csv files

# python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                         --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossV1"\
#                         --log_file="./logs/test_log_wav2vec2_NO_customLossV1_ASDmetric.txt"\
#                         --get_orig_model_results=0\
#                         --metric_to_use="asd_metric.py"\
#                         --extract_transcriptions=1

# python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                         --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossV1"\
#                         --log_file="./logs/test_log_wav2vec2_NO_customLossV1_WERmetric.txt"\
#                         --get_orig_model_results=0\
#                         --metric_to_use="wer"\
#                         --extract_transcriptions=0

# cd logs
# mkdir CustomLoss_V2
# cd ../

# python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                     --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_origLossV2"\
#                                     --log_file_name="test_log_wav2vec2_NO_origLossV2"\
#                                     --log_dir="./logs/CustomLoss_V2/"\
#                                     --get_orig_model_results=1\

cd logs
mkdir customLossTRIAL26_titan1
cd ../

python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossTRIAL26_titan1"\
                                    --log_file_name="test_log_wav2vec2_NO_customLossTRIAL26_titan1"\
                                    --log_dir="./logs/customLossTRIAL26_titan1/"\
                                    --get_orig_model_results=0\


# for n in {2..5}
# do
#     echo TESTING MODEL CUSTOM LOSS w/ LAMBDA 0.$n

#     cd logs/lambda_trials/
#     mkdir CustomLoss_V8_lambda0p${n}
#     cd ../../

#     python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                         --fine_tuned_model="../../fine_tuned_models/lambda_trials/wav2vec2_NO_customLossV8_lambda_0$n"\
#                                         --log_file_name="test_log_wav2vec2_NO_customLossV8_lambda_0$n"\
#                                         --log_dir="./logs/lambda_trials/CustomLoss_V8_lambda0p${n}/"\
#                                         --get_orig_model_results=0

# done
