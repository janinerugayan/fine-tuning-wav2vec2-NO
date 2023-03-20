cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_origLossV1
cd fine-tuning_wav2vec2_origLossV1
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_origLossV1
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customLoss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_origLossV1"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_origLossV1/"\
                                           --num_train_epochs=3\
                                           --learning_rate=1e-4\
                                           --use_asd_metric=0



cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_customLossV1
cd fine-tuning_wav2vec2_customLossV1
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_customLossV1
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_customLossV1"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_customLossV1/"\
                                           --num_train_epochs=3\
                                           --learning_rate=1e-4\
                                           --use_asd_metric=1
