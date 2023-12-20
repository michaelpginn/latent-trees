# latent-trees
Repository for **Hierarchical and Recursive Generalization in Language Models**

`data-generation.ipynb` is used to generate data using a context-free grammar.

`src/TreeTransformer` contains code for custom Tree Transformer variant of BERT model.

To train models, run
```
python3 train.py --dataset 'ID'|'GEN'|'GENX' [--pretrained] --train_epochs 1000
```

Adding `--pretrained` will use a pretrained BERT model.
