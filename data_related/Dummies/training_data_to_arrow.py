from datasets import load_dataset
from transformers import ElectraTokenizer
import os
import random

# pleas set maximum sequence length
MAX_SEQ_LEN = 512


class example_builder(object):
    def __init__(self, tokenizer, max_length, f_out):
        self._tokenizer = tokenizer
        self._current_sentences = []
        self._current_length = 0
        self._max_length = max_length
        self._target_length = max_length
        self.f_out = f_out

        self._current_sentences_origin = []

        self.cnt_examples = 0
        self.num_unk_tok = 0
        self.origin_line = ""

    def add_line(self, line):
        """Adds a line of text to the current example being built."""
        line = line.strip().replace("\n", " ")
        if (not line) and self._current_length != 0:  # empty lines separate docs
            boundary =  self._create_example()  # one document is finished -> creating examples
            self.f_out.write(' '.join(self._current_sentences_origin[:boundary+1]))
            if boundary > 0:
                self.f_out.write(" [SEP] ")
            self.f_out.write(' '.join(self._current_sentences_origin[boundary:]) +"\n")
            self._current_sentences_origin = []

        bert_tokens = self._tokenizer.tokenize(line)

        self._current_sentences.append(bert_tokens) 
        self._current_sentences_origin.append(line)
        self._current_length += len(bert_tokens) 

        if self._current_length >= self._target_length:
            boundary = self._create_example()
            self.f_out.write(' '.join(self._current_sentences_origin[:boundary + 1]))
            if boundary > 0:
                self.f_out.write(" [SEP] ")
            self.f_out.write(' '.join(self._current_sentences_origin[boundary:]) + "\n")
            self._current_sentences_origin = []

        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # small chance to only have one segment as in classification tasks
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 due to not yet having [CLS]/[SEP] tokens in the input text
            first_segment_target_length = (self._current_length - 3) // 2

        first_segment = []
        second_segment = []
        boundary = -1
        for i, sentence in enumerate(self._current_sentences):
            # the sentence goes to the first segment if
            # (1) the first segment is empty,
            # (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or len(first_segment) + len(sentence) < first_segment_target_length or
                    (len(second_segment) == 0 and len(
                        first_segment) < first_segment_target_length and random.random() < 0.5)):
                first_segment += sentence
            else:
                if boundary < 0 :
                    boundary = i
                second_segment += sentence

        # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length - len(first_segment) - 3)]

        # prepare to start building the next example
        self._current_sentences = []
        self._current_length = 0

        # small chance for random-length instead of max_length-length example
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        self.cnt_examples += 1
        """
        Richard Wang 은 아래와 같이 함수를 불러왔음
        return self._make_example(first_segment, second_segment)
        """
        return boundary 

    def _make_example(self, first_segment, second_segment):
        """Converts two "segments" of text into a tf.train.Example."""

        input_ids = first_segment + ["[SEP]"]
        if second_segment:
            input_ids += second_segment 

        self.cnt_examples += 1

        return input_ids 


def tokenize_function(examples):
    padding = "max_length" if MAX_SEQ_LEN else False
    # Remove empty lines

    return tokenizer(
        examples["text"],
        padding=padding,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


if __name__ == "__main__":
    # dir path for raw sentence datasets
    data_dir_path = "/Users/hmc/Desktop/NLP_DATA"
    # output dir path for new examples (segmentation data) 
    segment_out_dir = "/Users/hmc/Desktop/NLP_DATA"
    # tokenizer path
    # tokenizer_path = '/nlp/users/juaekim/my_electra_torch/airslm-base/'
   
    ####### 1. Create new data example with segment by maximum length
    # load tokenizer
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    # tokenizer = ElectraTokenizer.from_pretrained(tokenizer_path)
    # load raw text data
    data_names = os.listdir(data_dir_path)
    # data shuffle
    random.shuffle(data_names)
    # dataset names list for multiple data
    all_datasets = []

    for data_name in data_names:
        datasets_path = os.path.join(data_dir_path, data_name)
        datasets = load_dataset('text', data_files=[datasets_path])
        data_file_path = os.path.join(segment_out_dir, data_name)

        all_datasets.append(datasets_path)
        print(data_name + ": data_loaded")

        # output file for new example
        f_out = open(datasets_path, "w", encoding='utf-8')

        # make example 
        ex = example_builder(tokenizer, max_length=MAX_SEQ_LEN, f_out=f_out)
        lines = datasets['train']['text']
        
        for in_data in lines:
            ex.add_line(line=in_data) 
        print(f"examples {ex.cnt_examples}")

        f_out.close()

    ####### 2. Load new data examples and make as pyarrow file
    datasets = load_dataset('text', data_files=all_datasets, cache_dir=segment_out_dir)
    preprocessing_num_workers = 1
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=None,
        load_from_cache_file=False,  # not data_args.overwrite_cache,
    )
    tokenized_datasets.save_to_disk(segment_out_dir)
