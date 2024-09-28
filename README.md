This repo is built for Model ProAttUnet which focuses on recognizing protein secondary structures and related data in this study.

### Evaluate model

If you want to evaluate our model on different test sets, you could act as following steps:

- create data in proper format that the model requires and then put it into the directory **test sets**
- run the instructions in the command line: **python evaluation.py --model_name ./Best Models/ProAttUnet**

And you will see the evaluation results later. Surely, you could modify the file as you want to adapt different situations.

### Predict sequence

We also provide the **predict.py** file to predict secondary structures of another sequences. You can run instructions like : 

**python predict.py --fasta_file ./fasta_files/2MXA.fasta --model_name ./Best Models/ProAttUnet**

Here, we have provided 3 example fasta files. You can add more fasta files into the directory **fasta_files** for more predictions. The model will return an 8-state label corresponding to each residue.
