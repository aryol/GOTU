# Implementation for "Generalization on the Unseen, Logic Reasoning and Degree Curriculum"

This is an implementation for "Generalization on the Unseen, Logic Reasoning and Degree Curriculum" presented at ICML 2023 in the PyTorch framework. You can read our paper on [arxiv](https://arxiv.org/abs/2301.13105) or you can check [ICML's website](https://proceedings.mlr.press/v202/abbe23a.html).

### Contents:

- `token_transformer.py` contains the code of the Transformer model used in the experiments. 
- `models.py` contains the code for the rest of the models including MLP, mean-field, and random feaures model. 
- `utilities.py` includes some helper functions for the main file, e.g., computation of Fourier coefficients.  
- `main.py` is the main file for the experiments.
- `examples.py` contains the definitions of the tasks used in the paper.  
- `script.sh` includes the commands for generating the results presented in the paper. Note that the hyperparameters used for different tasks/models is also included in this file. 

## BibTex Citation
You can use the following citation for our paper. 

```
@InProceedings{pmlr-v202-abbe23a,
  title = 	 {Generalization on the Unseen, Logic Reasoning and Degree Curriculum},
  author =       {Abbe, Emmanuel and Bengio, Samy and Lotfi, Aryo and Rizk, Kevin},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {31--60},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/abbe23a/abbe23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/abbe23a.html},
}
```

