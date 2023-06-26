EXP_NUM="lightstage"
CUDA_NUM="0"
TESTING="False"
EXP_CODE="exp_01"
echo $EXP_NUM/cuda:$CUDA_NUM/Testing:$TESTING
python train.py --config configs/$EXP_NUM/helmet_front_left.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/knight_kneeling.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/knight_standing.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/knight_fighting.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/helmet_side_left.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/plant_left.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE