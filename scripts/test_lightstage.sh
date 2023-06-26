EXP_NUM="lightstage"
CUDA_NUM="0"
TESTING="True"
QUICK_TESTING="True"
EXP_CODE="exp"
echo $EXP_NUM/cuda:$CUDA_NUM/Testing:$TESTING
python test.py --config configs/$EXP_NUM/helmet_front_left.yml --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICK_TESTING --exp_code $EXP_CODE
python test.py --config configs/$EXP_NUM/knight_kneeling.yml --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICK_TESTING --exp_code $EXP_CODE
python test.py --config configs/$EXP_NUM/knight_standing.yml --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICK_TESTING --exp_code $EXP_CODE
python test.py --config configs/$EXP_NUM/knight_fighting.yml --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICK_TESTING --exp_code $EXP_CODE
python test.py --config configs/$EXP_NUM/helmet_side_left.yml --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICK_TESTING --exp_code $EXP_CODE
python test.py --config configs/$EXP_NUM/plant_left.yml --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICK_TESTING --exp_code $EXP_CODE