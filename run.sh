#CUDA_VISIBLE_DEVICES=0 python test_one_staged_lqr_pixel.py --config cheetah-run-embedlqr-state
# CUDA_VISIBLE_DEVICES=0,1 python test_one_staged_lqr_pixel.py --config cartpole-swingup-embedlqr-state
# CUDA_VISIBLE_DEVICES=0,1 python test_one_staged_lqr_pixel.py --config cartpole-swingup-embedlqr

xvfb-run -s "-screen 0 1400x900x24" python test_one_staged_lqr_pixel.py --config cheetah-run-embedlqr-state
#ssh -L 6006:127.0.0.1:6006 srr8@cs-mars-05.cmpt.sfu.ca
#xvfb-run -s "-screen 0 1400x900x24" python algorithms/learn_dynamic.py