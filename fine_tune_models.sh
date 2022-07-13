# cd ../model_ckpts/
# mkdir fine-tuning_wav2vec2_v12
# cd fine-tuning_wav2vec2_v12
# mkdir runs
# cd ../../fine_tuned_models/
# mkdir wav2vec2_NO_v12
# cd ../github/fine-tuning-wav2vec2-NO/

python fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                       --fine_tuned_model_ver="fine-tuning_wav2vec2_v12"\
                                       --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v12/"\
                                       --num_train_epochs=30\
                                       --learning_rate=1e-4

# python fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-1b-bokmaal"\
#                                       --fine_tuned_model_ver="fine-tuning_wav2vec2_v9"\
#                                       --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v9/"\
#                                       --num_train_epochs=4\
#                                       --learning_rate=2e-5
