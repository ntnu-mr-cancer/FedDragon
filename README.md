# FedDRAGON Algorithm

This repository implements simulated federated baselines for the federated arm of the DRAGON challenge (FedDRAGON). This repository is meant as a starting point to create and develop federated solutions. The base configuration implemented in this repository mimics the [centralized FedDRAGON BERT Base Mixed-domain baseline](https://github.com/ntnu-mr-cancer/feddragon_bert_base_mixed_domain). The config can be found in [configs/config.json](configs/config.json). 

The model is an adaption of the updated [DRAGON baseline](https://github.com/ntnu-mr-cancer/dragon_baseline) (version `0.4.6`), with pretrained foundational model `joeranbosma/dragon-bert-base-mixed-domain`.

For details on the pretrained foundational model, check out HuggingFace: [huggingface.co/joeranbosma/dragon-bert-base-mixed-domain](https://huggingface.co/joeranbosma/dragon-bert-base-mixed-domain).

The federated training was performed using the [Flower framework (1.17.0)](https://flower.ai/) with Federated averaging as the aggregation strategy.
## Development guide
It is strictly recommended to develop your own solutions in a docker container. Please use the provided docker file as a starting point for new solutions. The docker container uses an offline version of Hugging Face by default as this is required for running the container on the Grand Challenge platform during submission. To build a docker image with a pretrained model preloaded the [build.sh](build.sh) script is provided for convenience.

```sh
❯ ./build.sh --help 

Usage: ./build.sh [MODEL_NAME]

Builds the Docker image with the specified model name.
If no model is provided, defaults to: joeranbosma/dragon-bert-base-mixed-domain

Examples:
  ./build.sh joeranbosma/dragon-roberta-large-domain-specific
  ./build.sh                         # uses default model
```

### How to configure your own model
The [DRAGON baseline](https://github.com/ntnu-mr-cancer/dragon_baseline) is a wrapper around the [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer). Any valid argument to the Hugging Face [TrainingArguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) can be configured using the `model_kwargs` key of the [configuration file](configs/config.json). The number of federation rounds can be set in also be set in the [configuration file](configs/config.json). 

The implemented `CustomFedAvg` strategy can be used as a baseline for implementing other aggregation strategies.

### Testing the code
In the current implementation the `process.py` script is run by the submitted container. A sample script to test training and implemented in [test.sh](test.sh).

## Caveats
The data in the federate DRAGON challenge is all located in a single json file containing data from all clients. Data is split by this code in center specific training data prior to training as part of the training scripts.

In this implementation only training in federated. Predictions are done in a centralized manner even if data belongs to different clients. Additionally, prior to the federated training, scaler parameters and class information to correctly configure the models are gathered from the centralized dataset prior to creating center specific datasets. 

For the FedDragon challenge we only require the training to be federated. You are allowed to collect statistics and to get class labels from the dataset and pass this on to the clients. All solutions are required to have a corresponding github repository that should be made available to the challenge organizers. 

**References:**

[1] J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, H. Huisman, DRAGON Consortium. Large Language Models in Healthcare: DRAGON Performance Benchmark for Clinical NLP. To be submitted.

[2] J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, H. Huisman, DRAGON Consortium (2024). DRAGON Statistical Analysis Plan (v1.0). Zenodo. https://doi.org/10.5281/zenodo.10374512

[3] PENDING FedDRAGON paper.