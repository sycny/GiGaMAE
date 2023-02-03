python -u main_lp.py --dataset Cora   --lr 0.01  --ratio 0.2 --node_p 0.4  --edge_p 0.4  --tau 0.1 --hidden_num 64 --out_channels 512 --head 4  --gpu_num cuda:0 --seed 1 
python -u main_lp.py --dataset CiteSeer   --lr 0.01  --ratio 1 --node_p 0.4  --edge_p 0.4  --tau 0.1 --hidden_num 32 --out_channels 512 --head 8 --gpu_num cuda:0 --seed 1 --walk_length 10 --context_size 10 --l1_f 0 --l2_f 8 
python -u main_lp.py --dataset WikiCS   --lr 0.002 --node_p 0.2  --edge_p 0.2  --ratio 0.8 --gpu_num cuda:0  --tau 0.1 --hidden_num 64 --out_channels 512 --head 4  --lrdec_2 500 --seed 1
python -u main_lp.py --dataset Amazon-Computers  --lr 0.01  --node_p 0.1  --edge_p 0.2 --ratio 0.3   --tau 0.1 --hidden_num 64 --out_channels 512 --head 4 --seed 1 
python -u main_lp.py --dataset Amazon-Photo  --lr 0.01  --node_p 0.1  --edge_p 0.2 --ratio 0.5   --tau 0.1 --hidden_num 64 --out_channels 512 --head 4 --seed 1 
python -u main_lp.py --dataset Coauthor-Phy  --lr 0.01  --ratio 0.2  --node_p 0.4  --edge_p 0.4 --tau 0.1 --hidden_num 32 --out_channels 512 --head 12  --node2vec_batchsize 2000 --lrdec_2 500 --seed 1
python -u main_lp.py --dataset Coauthor-CS  --lr 0.02  --ratio 0.2  --node_p 0.4  --edge_p 0.4 --tau 0.1 --hidden_num 32 --out_channels 512 --head 12   --seed 1  
