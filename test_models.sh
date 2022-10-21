python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                        --fine_tuned_model="../../model_ckpts/fine-tuning_wav2vec2_v15_extracted/checkpoint-27500_extracted/"\
                        --log_file="./logs/test_log_wav2vec2_v15_checkpoint-27500_extracted.txt"\
                        --get_orig_model_results=0

# python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                         --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v15/"\
#                         --log_file="./logs/test_log_wav2vec2_v15.txt"\
#                         --get_orig_model_results=0

# python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-1b-bokmaal"\
#                         --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v9/"\
#                         --log_file="./logs/test_log_wav2vec2_v9.txt"\
#                         --get_orig_model_results=0

# python test_wav2vec2.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                         --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_v11/"\
#                         --log_file="./logs/test_log_wav2vec2_v11.txt"\
#                         --get_orig_model_results=0
