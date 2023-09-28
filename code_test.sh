cd ../../model_ckpts/
mkdir TEST_ONLY
cd TEST_ONLY
mkdir runs
cd ../../fine_tuned_models/
mkdir TEST_ONLY
cd ../github/fine-tuning-wav2vec2-NO/

# CUDA_VISIBLE_DEVICES=1
python3 fine_tuning_wav2vec2_customloss.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="TEST_ONLY"\
                                           --export_model_dir="../../fine_tuned_models/TEST_ONLY/"\
                                           --num_train_epochs=10\
                                           --learning_rate=1e-4\
                                           --lambda_asd=0.3\
                                           --use_asd_metric=1\
                                           --wandb_name="TEST_ONLY"