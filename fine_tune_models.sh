cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_v17
cd fine-tuning_wav2vec2_v17
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_v17
cd ../github/fine-tuning-wav2vec2-NO/

# python3 fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                        --fine_tuned_model_ver="fine-tuning_wav2vec2_v16"\
#                                        --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v16/"\
#                                        --num_train_epochs=12\
#                                        --learning_rate=1e-4


python fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-1b-bokmaal"\
                                      --fine_tuned_model_ver="fine-tuning_wav2vec2_v17"\
                                      --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v17/"\
                                      --num_train_epochs=12\
                                      --learning_rate=2e-5
