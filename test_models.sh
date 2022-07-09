python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v10/"\
                        --log_file="./logs/test_log_wav2vec2_v10.txt"\
                        --get_orig_model_results=True

python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v11/"\
                        --log_file="./logs/test_log_wav2vec2_v11.txt"\
                        --get_orig_model_results=False 
