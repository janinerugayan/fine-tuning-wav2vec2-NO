# for a in `seq 1 2 3`;
# do

#     mkdir -p ../../model_ckpts/fine-tuning_wav2vec2_asd_trial77_mwer_6ep_lambda0p${a}_allDataSmall_aulus6/runs
#     mkdir -p ../../fine_tuned_models/wav2vec2_NO_asd_trial77_mwer_6ep_lambda0p${a}_allDataSmall_aulus6

#     python3 fine_tuning_wav2vec2_customlossV3.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                                --fine_tuned_model_ver="fine-tuning_wav2vec2_asd_trial77_mwer_6ep_lambda0p${a}_allDataSmall_aulus6"\
#                                                --export_model_dir="../../fine_tuned_models/wav2vec2_NO_asd_trial77_mwer_6ep_lambda0p${a}_allDataSmall_aulus6/"\
#                                                --num_train_epochs=6\
#                                                --learning_rate=1e-4\
#                                                --lambda_asd=0.${a}\
#                                                --use_asd_metric=1\
#                                                --use_asd_scores=0\
#                                                --wandb_name="asd_trial77_mwer_6ep_lambda0p${a}_allDataSmall_aulus6"\
#                                                --export_log="./loss_logs/trial77_mwer_6ep_lambda0p${a}_allDataSmall_aulus6.txt"

# done

mkdir -p ./model_ckpts/NOtrial_frompretrained_origlossv1_10ep_deepthought/runs
mkdir ./fine_tuned_models/NOtrial_frompretrained_origlossv1_10ep_deepthought
python3 fine_tuning_wav2vec2_frompretrained.py --original_model="facebook/wav2vec2-base-100h"\
                                        --fine_tuned_model_ver="NOtrial_frompretrained_origlossv1_10ep_deepthought"\
                                        --export_model_dir="./fine_tuned_models/NOtrial_frompretrained_origlossv1_10ep_deepthought/"\
                                        --num_train_epochs=10\
                                        --learning_rate=1e-4\
                                        --use_asd_metric=0\
                                        --lambda_asd=0.3\
                                        --num_paths=3\
                                        --normalized_score=0\
                                        --wandb_name="NOtrial_frompretrained_origlossv1_10ep_deepthought"\
                                        --from_checkpoint=0\
                                        --checkpoint_path="./model_ckpts/NOtrial_frompretrained_origlossv1_10ep_deepthought/checkpoint-2734"\
                                        --training_data="load_csv"