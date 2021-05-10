from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Optional

from transformers import (
    HfArgumentParser,
    RobertaTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments
)

from tag_mspan_robert_gcn.roberta_batch_gen_tmspan import DropDataset, DropDataCollator
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import NumericallyAugmentedBertNet
from tools.utils import DropEmAndF1


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    dropout: float = field(default=0)
    use_gcn: bool = field(default=False)
    gcn_steps: int = field(default=0)


@dataclass
class DataTrainingArguments:
    data_dir: str
    max_train_samples: Optional[int] = field(default=None)


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)

    logger.info("Loading data...")
    train_dataset = DropDataset(data_args, data_mode="train", tokenizer=tokenizer)
    eval_dataset = DropDataset(data_args, data_mode="dev", tokenizer=tokenizer)

    logger.info("Building Roberta model...")
    roberta_model = RobertaModel.from_pretrained(model_args.model_name_or_path)

    logger.info("Building Drop model...")
    model = NumericallyAugmentedBertNet(roberta_model,
                                        hidden_size=roberta_model.config.hidden_size,
                                        dropout_prob=model_args.dropout,
                                        use_gcn=model_args.use_gcn,
                                        gcn_steps=model_args.gcn_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DropDataCollator(tokenizer),
        compute_metrics=None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

if __name__ == '__main__':
    main()
