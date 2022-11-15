cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_trialcustomloss
cd fine-tuning_wav2vec2_trialcustomloss
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_trialcustomloss
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_trialcustomloss"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_trialcustomloss/"\
                                           --num_train_epochs=1\
                                           --learning_rate=1e-4\
                                           --use_asd_metric=1
