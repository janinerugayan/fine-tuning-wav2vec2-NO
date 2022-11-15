python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v18/"\
                        --log_file="./logs/test_log_wav2vec2_v18_ASDmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="asd_metric.py"\
                        --extract_transcriptions=1

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v18/"\
                        --log_file="./logs/test_log_wav2vec2_v18_WERmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="wer"\
                        --extract_transcriptions=0

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v19/"\
                        --log_file="./logs/test_log_wav2vec2_v19_ASDmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="asd_metric.py"\
                        --extract_transcriptions=1

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v19/"\
                        --log_file="./logs/test_log_wav2vec2_v19_WERmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="wer"\
                        --extract_transcriptions=0
