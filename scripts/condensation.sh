# cora
# 025
python condense_transduct_2.py --section cora-r1  --gpuid 0 --lam 0.75 --T 1500 --scheduler geom  \
--min_start_epoch 0 --max_start_epoch 200 --expert_epochs 1500   \
--syn_steps 2500 --max_start_epoch_s 50 --lr_feat 0.0001 --lr_y 0.00005 --beta 0.01 --soft_label

# 05
python condense_transduct_2.py --section cora-r05  --gpuid 0 --lam 0.75 --T 1500 --scheduler geom   \
--min_start_epoch 0 --max_start_epoch 200 --expert_epochs 1400   \
--syn_steps 2500 --max_start_epoch_s 50 --lr_feat 0.0001 --lr_y 0.00005 --beta 0.01 --soft_label

# 1
python condense_transduct_2.py --section cora-r025  --gpuid 0 --lam 0.75 --T 1500 --scheduler geom   \
--min_start_epoch 0 --max_start_epoch 200 --expert_epochs 1400   \
--syn_steps 2500 --max_start_epoch_s 50 --lr_feat 0.0001 --lr_y 0.00005 --beta 0.01 --soft_label



# arxiv
# 01
python condense_transduct_1.py --section ogbn-arxiv-r01  --gpuid 0 --lam 0.85 --T 1200 --scheduler root  \
--min_start_epoch 0 --max_start_epoch 400 --expert_epochs 1750 --syn_steps 2000  \
--max_start_epoch_s 350 --lr_feat 0.05 --lr_y 0.001 

# 005
python condense_transduct_1.py --T=1200 --beta=0 --max_start_epoch_s=200 --expert_epochs=1650 --lam=0.85   \
--lr_feat=0.03 --lr_y=0.001 --max_start_epoch=350 --min_start_epoch=0 --scheduler=root   \
--section=ogbn-arxiv-r005  --syn_steps=2200

# 001
python condense_transduct_1.py --T=1200 --beta=0 --max_start_epoch_s=200 --expert_epochs=1900 --lam=0.85  \
--lr_feat=0.03 --lr_y=0.001 --max_start_epoch=350 --min_start_epoch=0 --scheduler=root   \
--section=ogbn-arxiv-r001  --syn_steps=2100

# 0005
python condense_transduct_1.py --T=1200 --beta=0 --max_start_epoch_s=200 --expert_epochs=1600   \
--lam=0.85 --lr_feat=0.05 --lr_y=0.001 --max_start_epoch=300 --min_start_epoch=0 --scheduler=root   \
--section=ogbn-arxiv-r0005  --syn_steps=2100

# 0001
python condense_transduct_2.py --T=1200 --beta=0 --max_start_epoch_s=30 --expert_epochs=1100 --lam=0.85   \
--lr_feat=0.25 --lr_y=0.001 --max_start_epoch=50 --min_start_epoch=0 --scheduler=root   \
--section=ogbn-arxiv-r0001  --syn_steps=650


# reddit
# 0125
python condense_induct_1.py --section reddit-r0125  --gpuid 0 --lam 0.9 --T 800 --scheduler linear  \
--min_start_epoch 0 --max_start_epoch 200 --syn_steps 1300 --expert_epochs 1100 --max_start_epoch_s 1   \
--lr_feat 0.001 --beta 0.25 --lr_y 0.0001 --tem 1.5 --lr_tem 0.005 --maxtem 2.5 --soft_label

# 0075
python condense_induct_1.py --T=800 --beta=0.2 --max_start_epoch_s=50 --expert_epochs=900     \
--lam=0.9 --lr_feat=0.001 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=250 --maxtem=1 --min_start_epoch=0 --scheduler=linear     \
--section=reddit-r0075  --syn_steps=1300 --tem=1 --soft_label

# 005
python condense_induct_1.py --T=800 --beta=0.2 --max_start_epoch_s=50 --expert_epochs=900     \
--lam=0.9 --lr_feat=0.001 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=250 --maxtem=1 --min_start_epoch=0 --scheduler=linear     \
--section=reddit-r005  --syn_steps=1300 --tem=1 --soft_label

# 003
python condense_induct_1.py --T=800 --beta=0.2 --max_start_epoch_s=50 --expert_epochs=900     \
--lam=0.9 --lr_feat=0.001 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=250 --maxtem=1 --min_start_epoch=0 --scheduler=linear     \
--section=reddit-r003  --syn_steps=1300 --tem=1 --soft_label

# 0005
python condense_induct_1.py --T=800 --beta=0.25 --max_start_epoch_s=1    \
--expert_epochs=900 --gpuid=2 --lam=0.9 --lr_feat=0.2 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=10 --maxtem=1 --min_start_epoch=0    \
--scheduler=linear --section=reddit-r0005  --syn_steps=800 --tem=1

# 0002
python condense_induct_1.py --T=800 --beta=0.25 --max_start_epoch_s=1    \
--expert_epochs=900 --gpuid=2 --lam=0.9 --lr_feat=0.2 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=10 --maxtem=1 --min_start_epoch=0    \
--scheduler=linear --section=reddit-r0002  --syn_steps=800 --tem=1

# 0001
python condense_induct_1.py --T=800 --beta=0.1 --max_start_epoch_s=1 --expert_epochs=1000 --gpuid=3 --lam=0.9   \
--lr_feat=0.03 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=20 --maxtem=1 --min_start_epoch=0 --scheduler=linear   \
--section=reddit-r0001  --syn_steps=1000 --tem=1

# 00005
python condense_induct_1.py --T=800 --beta=0.25 --max_start_epoch_s=1 --expert_epochs=800 --gpuid=1 --lam=0.9  \
--lr_feat=0.02 --lr_tem=0.005 --lr_y=0.0001 --max_start_epoch=50 --maxtem=1 --min_start_epoch=0   \
--scheduler=linear --section=reddit-r00005  --syn_steps=800 --tem=1


# citeseer
# 025
python condense_transduct_2.py --T=250 --beta=0.1 --max_start_epoch_s=1 --expert_epochs=350 --gpuid=1    \
--lam=0.8 --lr_feat=0.0005 --lr_y=5e-05 --max_start_epoch=20 --min_start_epoch=0 --scheduler=root --section=citeseer-r1  --syn_steps=400

# 05
python condense_transduct_2.py --T=250 --beta=0.05 --max_start_epoch_s=10 --expert_epochs=350 --gpuid=1     \
--lam=0.8 --lr_feat=0.0007 --lr_y=5e-05 --max_start_epoch=80 --min_start_epoch=0 --scheduler=root --section=citeseer-r05  --syn_steps=200

# 1
python condense_transduct_2.py --T=250 --beta=0.1 --max_start_epoch_s=1 --expert_epochs=350 --gpuid=1     \
--lam=0.8 --lr_feat=0.0001 --lr_y=5e-05 --max_start_epoch=30 --min_start_epoch=0 --scheduler=root --section=citeseer-r025  --syn_steps=200

# flickr
# 001
python condense_induct_1.py --T=100 --beta=0.3    \
--max_start_epoch_s=10 --expert_epochs=70 --lam=0.95   \
--lr_feat=0.07 --lr_y=0.001 --max_start_epoch=70 --min_start_epoch=0 --scheduler=root    \
--section=flickr-r001  --syn_steps=300

# 0005
python condense_induct_1.py --T=100 --beta=100 --max_start_epoch_s=30 --expert_epochs=600     \
--lam=0.95 --lr_feat=0.1 --lr_y=0.001 --max_start_epoch=60 --min_start_epoch=0 --scheduler=root     \
--section=flickr-r0005  --syn_steps=200

# 0001
python condense_induct_1.py --T=150 --beta=0.3 --max_start_epoch_s=10 --expert_epochs=600      \
--lam=0.95 --lr_feat=0.07 --lr_y=0.001 --max_start_epoch=30 --min_start_epoch=0 --scheduler=root      \
--section=flickr-r0001  --syn_steps=700
