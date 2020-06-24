# JM-NAT

Code for our ACL 2020 [paper](https://www.aclweb.org/anthology/2020.acl-main.36.pdf), "Jointly Masked Sequence-to-Sequence Model for Non-Autoregressive Neural Machine Translation". Please cite our paper if you find this repository helpful in your research:
```
@inproceedings{guo2020jointly,
    title = {Jointly Masked Sequence-to-Sequence Model for Non-Autoregressive Neural Machine Translation},
    author = {Guo, Junliang and Xu, Linli and Chen, Enhong},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year = {2020},
    publisher = {Association for Computational Linguistics},
    pages = {376--385},
}
```

## Requirements

The code is based on [fairseq-0.6.2](https://github.com/pytorch/fairseq/tree/v0.6.2), PyTorch-1.2.0 and cuda-9.2.

## Training Steps

To train a non-autoregressive machine translation model, please follow the three steps listed below:

* Firstly, follow the instructions in [fairseq](https://github.com/pytorch/fairseq) to train an autoregressive model.
* Generate distilled target samples by the autoregressive model, i.e., set `--gen-subset train` while decoding.
* Train our model on the distilled training set. For example, on the IWSLT14 De-En task:
```
python train.py $DATA_DIR \
  --task xymasked_seq2seq \
  -a transformer_nat_ymask_pred_len_deep_small --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --lr 0.0007 --min-lr '1e-09' \
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.0 --max-tokens 4096 --max-update 500000 --mask-source-rate 0.1
```

While inference, our model utilizes similar decoding algorithm proposed in [Mask-Predict](https://github.com/facebookresearch/Mask-Predict), and we use the average of last 10 checkpoints to obtain the results:
```
python generate.py $DATA_DIR \
  --task xymasked_seq2seq --path checkpoint_aver.pt --mask_pred_iter 10 \
  --batch-size 64 --beam 4 --lenpen 1.1 --remove-bpe
```
