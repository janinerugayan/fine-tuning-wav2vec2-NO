python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v16/"\
                        --log_file="./logs/test_log_wav2vec2_v16.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="wer"

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v15/"\
                        --log_file="./logs/test_log_wav2vec2_v15_ASDmetric.txt"\
                        --get_orig_model_results=1\
                        --metric_to_use="asd_metric.py"

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v16/"\
                        --log_file="./logs/test_log_wav2vec2_v16_ASDmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="asd_metric.py"

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v13/"\
                        --log_file="./logs/test_log_wav2vec2_v13_ASDmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="asd_metric.py"

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v14/"\
                        --log_file="./logs/test_log_wav2vec2_v14_ASDmetric.txt"\
                        --get_orig_model_results=0\
                        --metric_to_use="asd_metric.py"
