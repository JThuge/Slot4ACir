CUDA_VISIBLE_DEVICES=1 \
python blip2_fine_tune.py \
--exp-name train_slot4acir_fiq \
--dataset fashioniq \
--blip-model-name blip2_light_cir \
--num-epochs 30 \
--num-workers 4 \
--learning-rate 2e-5 \
--batch-size 96 \
--llm-sampler \
--save-training \
--save-best \
--validation-frequency 1

# CUDA_VISIBLE_DEVICES=0 \
# python blip2_fine_tune.py \
# --exp-name train_slot4acir_cirr \
# --dataset cirr \
# --blip-model-name blip2_light_cir \
# --num-epochs 50 \
# --num-workers 4 \
# --learning-rate 1e-5 \
# --batch-size 96 \
# --llm-sampler \
# --save-training \
# --save-best \
# --validation-frequency 1

# optional
# --light-model-name efficientnet-b0 by default (choose from ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientvit-m4", "efficientvit-m2", "mobilenetv3-s", "mobilenetv3-l"])
# --num-slots 8 by default can be 16, 32...
# --use-adapt (whether to use adaptive slot attention [ref: Adaptive Slot Attention: Object Discovery with Dynamic Slot Number, Fan et al.])
# --loss-setting itc dta by default (nargs='+', must contains itc)