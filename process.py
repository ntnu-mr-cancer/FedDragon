from dragon_baseline import DragonBaseline


class DragonRobertaLargeDomainSpecificV2(DragonBaseline):
    def __init__(self, **kwargs):
        """
        Adapt the DRAGON baseline to use the joeranbosma/dragon-roberta-large-domain-specific model.
        Note: when manually changing the model, update the Dockerfile to pre-download that model.
        """
        super().__init__(**kwargs)
        self.model_name = "joeranbosma/dragon-roberta-large-domain-specific"
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 8
        self.gradient_checkpointing = False
        self.max_seq_length = 512
        self.learning_rate = 1e-05
        self.num_train_epochs = 5
        self.create_strided_training_examples = True


if __name__ == "__main__":
    DragonRobertaLargeDomainSpecificV2().process()
