# Info
  - For computational efficiency, I aim to train BERT-SMALL(described by [ELECTRA, ICLR 2020](https://arxiv.org/pdf/2003.10555.pdf)) 
    and I will use ELECTRA FRAMEWORK rather than vanilla BERT MLM task for pretraining
  - I referred to [richarddwang's repository](https://github.com/richarddwang/electra_pytorch) for implementation.  
  - Ultimately, I will point out which are good hyperparameters for BERT's series at least BERT-SMALL case.
  - In brief, this repository aims to do the implementation of ELECTRA.
# About ELECTRA
  - As described in [BERT paper](https://arxiv.org/pdf/1810.04805.pdf), BERT should be trained in two steps.
  - First, pretrain a BERT model with two tasks denoted by masked language model and next sentence prediction.
  - Second, fine-tune the pretrained BERT model for each task. 
    - Since I just want to check the benchmark score for some of my other task, in this repository, I only provide the related code base
    and information.
  - As pointed out in [ELECTRA, ICLR 2020](https://arxiv.org/pdf/2003.10555.pdf), It is inefficient way of pretraining
    a BERT model.
  - n this repository, pretraining process follows ELECTRA framework due to efficiency and performance.

# Requirements
  - pytorch 1.7+, numpy, python 3.7, tqdm, transformers
# Features
  - Current version provides only 1-GPU training.
# Dataset
  - Pretraining : English wikipedia, Bookscorpus
  - Before training, you should download above two datasets and convert those a one 
txt format dataset. The converted dataset must be aligned sentence by sentence by using \n.
    For example, 
    - *I love you so much. \n*
    - *The pig walks awy from this farm. \n*
    - ...
    - *Tesla stock is going to be 2,000 dollars \n*
# Usage
  - For pretraining
    - ```
      python Pretraining.py 
      ```
  - For fine-tuning
    - will be updated
  - For LM_DEMO
    - When you want to test your model whether it is working well or not, use LM_DEMO.py.
      - First, make a txt file consisted of sentences that you want to change.
      - Second, give the path of weight file for generator as an argument when you run the LM_DEMO.py script.
      - Run the LM_DEMO.py script as follows.
        ```python3
        python LM_DEMO.py
        ```
        
# Curruent Status
  - Finetuning dataloader is updated.
  - Add-on module will be added for downstream task belonging to GLUE benchmark.
# To do
  - Update for usage description of fine-tuning script. 
# Miscellaneous
  - Though the ELECTRA paper's author described that they didn't back-propagate the discriminator loss 
    through the generator due to sampling step, actually, we can back-propagate by using gumbel softmax.
    But, according to [richarddwang's repository](https://github.com/richarddwang/electra_pytorch), 
    I remove the gradient graph for the sampling parts. (I used gumbel softmax provided from pytorch with minor 
    modification due to [a bug](https://github.com/richarddwang/electra_pytorch))