python fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-1b-bokmaal"\
                                       --fine_tuned_model_ver="fine-tuning_wav2vec2_v8"\
                                       --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v8/"\
                                       --num_train_epochs=12\
                                       --learning_rate=2e-5

python fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-1b-bokmaal"\
                                      --fine_tuned_model_ver="fine-tuning_wav2vec2_v9"\
                                      --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v9/"\
                                      --num_train_epochs=4\
                                      --learning_rate=2e-5
