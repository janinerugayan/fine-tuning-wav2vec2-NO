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

python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_origLossV1"\
                                    --log_file_name="test_log_wav2vec2_NO_origLossV1"\
                                    --get_orig_model_results=0\

python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossV1"\
                                    --log_file_name="test_log_wav2vec2_NO_customLossV1"\
                                    --get_orig_model_results=0\
