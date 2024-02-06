#@citeseer-buff-cl
CUDA_VISIBLE_DEVICES=3 python buffer_transduct_cl.py --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset citeseer   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam --lam 0.8 --T 250 --scheduler root

#@cora-buff-cl
CUDA_VISIBLE_DEVICES=4 python buffer_transduct_cl.py --lr_teacher 0.4 \
--teacher_epochs 3000 --dataset cora   \
--num_experts=200 --wd_teacher 0 --optim SGD --lam 0.75 --T 1500 --scheduler geom

#@ogbn-buff-cl
CUDA_VISIBLE_DEVICES=2 python buffer_transduct_cl.py --lr_teacher 1 \
--teacher_epochs 2400 --dataset ogbn-arxiv   \
--num_experts=200 --wd_teacher 0 --optim SGD --lam 0.85 --T 1200 --scheduler root

#@flickr-buff-cl
CUDA_VISIBLE_DEVICES=5 python buffer_inductive_cl.py --lr_teacher 0.001 \
--teacher_epochs 1100 --dataset flickr   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam --lam 0.95 --T 1200 --scheduler root

#@reddit-buff-cl
CUDA_VISIBLE_DEVICES=4 python buffer_inductive_cl.py --lr_teacher 0.001 \
--teacher_epochs 1000 --dataset reddit   \
--num_experts=200 --wd_teacher 5e-4 --optim Adam --lam 0.9 --T 800 --scheduler linear


