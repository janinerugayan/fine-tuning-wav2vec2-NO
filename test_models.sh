# fine-tuned model path should be withouth "/" at the end of the path

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossV1"\
                        --log_file="./logs/test_log_wav2vec2_NO_customLossV1_ASDmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="asd_metric.py"\
                        --extract_transcriptions=1

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossV1"\
                        --log_file="./logs/test_log_wav2vec2_NO_customLossV1_WERmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="wer"\
                        --extract_transcriptions=0
