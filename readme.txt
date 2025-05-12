python=3.8.18
torch=1.8.1+cu111
dgl-cu111=0.6.1
scikit-learn = 1.3.2
scipy = 1.8.0
tqdm = 4.66.1
numpy = 1.22.4

For FB15K and TMDB5K:
xx_rel.pk: graph data and structure features
xx_lang.pk: semantic features

For IMDB:
imdb_s_rel_150000.pk: graph data
imdb_s_node2vec_new_150000.pk: structure features
imdb_s_lang_150000.pk: semantic features

dataset link: https://drive.google.com/drive/folders/1BHUuUo7WOmNOYvdvN7r8iIebp_e-Npr3?usp=sharing
You only need to download the raw_datasets


=======step 1=======
split labels:
=======step 1=======

1. python process_datasets/split.py --dataset_name FB15K --raw_data_path {your_raw_datasets} --new_data_path {your_new_datasets}
Furthermore, you could set the train_ratio and the random seed in process_datasets/split.py
dataset_name: Can be FB15k, TMDB5K, IMDB

raw_data_path: the path of raw datasets. It should be:
├── raw_datasets
│   ├── FB15k
│   │   ├── fb_lang.pk
│   │   ├── fb15k_rel.pk
│   │   └── ...
│   ├── TMDB5K
│   └── ...
│   ├── IMDB
│   └── ...
└── ...    

new_data_path: the path of new data after split. It would be:
├── new_datasets
│   ├── FB15k
│   │   ├── dataset_split
│   │   │   ├── idx_1000
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── fb_lang.pk
│   │   ├── fb15k_rel.pk
│   │   └── ...
│   ├── TMDB5K
│   └── ...
│   ├── IMDB
│   └── ...
└── ...    

We split the labels and save them in the dataset_split.

=======step 2=======
train model and evaluate
=======step 2=======

Firstly, you need to set the dataset path:

FB15K: python easing/main_easing.py --dataset FB15K \
--data_path {your path} \
--graph_data FB15K/fb15k_rel.pk --semantic_data FB15K/fb_lang.pk \
--residual True --norm True --spm True --train_num 1.0 \
--samp_ssl 5 --w_ulb 1.0 --loss_beta 0.0 --centrality_gamma 0.9 \
--REnt_bool True --unc_layers 2

TMDB15K: python easing/main_easing.py --dataset TMDB5K \
--data_path {your path} \
--graph_data TMDB5K/tmdb_rel.pk --semantic_data TMDB5K/tmdb_lang.pk \
--residual True --norm True --spm True --train_num 1.0 \
--samp_ssl 5 --w_ulb 1.0 --loss_beta 0.0 --centrality_gamma 0.9 \
--REnt_bool True --unc_layers 1

IMDB: python easing/main_easing.py --dataset IMDB \
--data_path {your path} \
--graph_data IMDB/imdb_s_rel_150000.pk --semantic_data IMDB/imdb_s_lang_150000.pk \
--structure_data IMDB/imdb_s_node2vec_new_150000.pk \
--residual True --norm True --spm True --train_num 1.0 \
 --samp_ssl 5 --w_ulb 1.0 --loss_beta 0.0 --centrality_gamma 0.6 \
 --REnt_bool True --unc_layers 1



