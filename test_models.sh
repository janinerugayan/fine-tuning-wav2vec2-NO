# fine-tuned model path should be WITHOUT "/" at the end of the path in order to save transcriptions in unique csv files

# cd logs
# mkdir customLossTRIAL46_sampledHinge_aulus5
# cd ../

# python test_wav2vec2_scoreperutt.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
#                                     --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_customLossTRIAL46_sampledHinge_aulus5"\
#                                     --log_file_name="test_log_wav2vec2_NO_customLossTRIAL46_sampledHinge_aulus5"\
#                                     --log_dir="./logs/customLossTRIAL46_sampledHinge_aulus5/"\
#                                     --get_orig_model_results=0

cd logs
mkdir expectedASD4_3ep_0p5_Rundkast_aulus6
echo expectedASD4_3ep_0p5_Rundkast_aulus6
cd ../

python test_wav2vec2_scoreperutt_OneData.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_expectedASD4_3ep_0p5_Rundkast_aulus6"\
                                    --log_file_name="test_log_wav2vec2_NO_expectedASD4_3ep_0p5_Rundkast_aulus6"\
                                    --log_dir="./logs/expectedASD4_3ep_0p5_Rundkast_aulus6/"\
                                    --get_orig_model_results=0

cd logs
mkdir expectedASD4_3ep_0p1_Rundkast_aulus6
echo expectedASD4_3ep_0p1_Rundkast_aulus6
cd ../

python test_wav2vec2_scoreperutt_OneData.py --original_model="NbAiLab/nb-wav2vec2-300m-bokmaal"\
                                    --fine_tuned_model="../../fine_tuned_models/wav2vec2_NO_expectedASD4_3ep_0p1_Rundkast_aulus6"\
                                    --log_file_name="test_log_wav2vec2_NO_expectedASD4_3ep_0p1_Rundkast_aulus6"\
                                    --log_dir="./logs/expectedASD4_3ep_0p1_Rundkast_aulus6/"\
                                    --get_orig_model_results=0