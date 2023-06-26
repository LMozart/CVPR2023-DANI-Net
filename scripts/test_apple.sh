EXP_NUM="apple"
CUDA_NUM="0"
TESTING="True"
QUICK_TESTING="True"
EXP_CODE="exp"
echo $EXP_NUM/cuda:$CUDA_NUM/Testing:$TESTING
python test.py --config configs/$EXP_NUM/apple.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE --quick_testing $QUICK_TESTING
python test.py --config configs/$EXP_NUM/gourd1.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE --quick_testing $QUICK_TESTING
python test.py --config configs/$EXP_NUM/gourd2.yml --cuda $CUDA_NUM   --testing $TESTING --exp_code $EXP_CODE --quick_testing $QUICK_TESTING