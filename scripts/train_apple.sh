EXP_NUM="apple"
CUDA_NUM="0"
TESTING="False"
EXP_CODE="exp_01"
echo $EXP_NUM/cuda:$CUDA_NUM/Testing:$TESTING
python train.py --config configs/$EXP_NUM/apple.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/gourd1.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE
python train.py --config configs/$EXP_NUM/gourd2.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE