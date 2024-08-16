cd ../../model_ckpts/
mkdir fine-tuning_wav2vec2_expectedASD4_3ep_inter1_Rundkast_aulus6
cd fine-tuning_wav2vec2_expectedASD4_3ep_inter1_Rundkast_aulus6
mkdir runs
cd ../../fine_tuned_models/
mkdir wav2vec2_NO_expectedASD4_3ep_inter1_Rundkast_aulus6
cd ../github/fine-tuning-wav2vec2-NO/

python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                           --fine_tuned_model_ver="fine-tuning_wav2vec2_expectedASD4_3ep_inter1_Rundkast_aulus6"\
                                           --export_model_dir="../../fine_tuned_models/wav2vec2_NO_expectedASD4_3ep_inter1_Rundkast_aulus6/"\
                                           --num_train_epochs=3\
                                           --learning_rate=1e-4\
                                           --lambda_asd=1\
                                           --use_asd_metric=1\
                                           --wandb_name="expectedASD4_3ep_inter1_Rundkast_300m_batched_aulus6"\
                                           --export_log="./loss_logs/expectedASD4_3ep_inter1_Rundkast_aulus6.txt"