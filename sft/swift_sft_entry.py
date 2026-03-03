import sys
from transformers import TrainingArguments, Seq2SeqTrainingArguments

# transformers main removed past_index; ms-swift trainer still references it.
if not hasattr(TrainingArguments, "past_index"):
    TrainingArguments.past_index = -1
if not hasattr(Seq2SeqTrainingArguments, "past_index"):
    Seq2SeqTrainingArguments.past_index = -1

from swift.llm.train.sft import sft_main

if __name__ == "__main__":
    sft_main()
