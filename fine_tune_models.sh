cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_v20
cd fine-tuning_wav2vec2_v20
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_v20
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                       --fine_tuned_model_ver="fine-tuning_wav2vec2_v20"\
                                       --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v20/"\
                                       --num_train_epochs=6\
                                       --learning_rate=1e-4\
                                       --use_asd_metric=0

cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_v21
cd fine-tuning_wav2vec2_v21
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_v21
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                       --fine_tuned_model_ver="fine-tuning_wav2vec2_v21"\
                                       --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v21/"\
                                       --num_train_epochs=6\
                                       --learning_rate=1e-4\
                                       --use_asd_metric=1


# python fine_tuning_wav2vec2_working.py --original_model="NbAiLab/nb-wav2vec2-1b-bokmaal"\
#                                       --fine_tuned_model_ver="fine-tuning_wav2vec2_v"\
#                                       --export_model_dir="../../fine_tuned_models/wav2vec2_NO_v/"\
#                                       --num_train_epochs=12\
#                                       --learning_rate=2e-5
