import os
import json
import torch
from tree_hugger.core import PythonParser
import torch.nn as nn
from model import Seq2Seq
from pathlib import Path
from utils import Example, convert_examples_to_features
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
## We are defining all the needed functions here. 
def inference(data, model, tokenizer):
    # Calculate bleu
    eval_sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=len(data))

    model.eval()
    p = []
    for batch in eval_dataloader:
        batch = tuple(t.to('cpu') for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[: t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    return (p, source_ids.shape[-1])


def get_features(examples, tokenizer):
    features = convert_examples_to_features(
        examples, tokenizer, stage="test"
    )
    all_source_ids = torch.tensor(
        [f.source_ids[: 256] for f in features], dtype=torch.long
    )
    all_source_mask = torch.tensor(
        [f.source_mask[: 256] for f in features], dtype=torch.long
    )
    return TensorDataset(all_source_ids, all_source_mask)


def build_model(model_class, config, tokenizer):
    encoder = model_class(config=config)
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=config.hidden_size, nhead=config.num_attention_heads
    )
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        config=config,
        beam_size=10,
        max_length=128,
        sos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
    )

    model.load_state_dict(
        torch.load(
            "pytorch_model.bin",
            map_location=torch.device("cpu"),
        ),
        strict=False,
    )
    return model
config = RobertaConfig.from_pretrained("microsoft/codebert-base")
tokenizer = RobertaTokenizer.from_pretrained(
    "microsoft/codebert-base", do_lower_case=False
)

model = build_model(
    model_class=RobertaModel, config=config, tokenizer=tokenizer
).to('cpu')
def check_out_path(target_path: Path):
    """"
    This function recursively yields all contents of a pathlib.Path object
    """
    yield target_path
    for file in target_path.iterdir():
        if file.is_dir():
            yield from check_out_path(file)
        else:
            yield file.absolute()


def is_python_file(file_path: Path):
  """
  This little function will help us to filter the result and keep only the python files
  """
  return file_path.is_file() and file_path.suffix == ".py"
pp = PythonParser(library_loc="C:/Nish/CodeSummary UI/my-languages.so")
def summarize_code():
    for file_path in check_out_path(Path("files-for-code")):
        m=[]
        if is_python_file(file_path):
        # we use one line, super convinient tree-hugger API call to get the needed data
            if pp.parse_file(str(file_path)):
                temp_cache = []
                # The following call returns a dict where each key is a name of a function
                # And each value is a tuple, (function_body, function_docstring)
                func_and_docstr = pp.get_all_function_bodies(strip_docstr=True)
                for func_name, (body, docstr) in func_and_docstr.items():
                    example = [Example(source=body, target=None)]
                    message, length = inference(get_features(example, tokenizer), model, tokenizer)
                    m.append(message)
    return m

def summarize_file():
    with open('summary.txt', 'w') as summary:
        for file_path in check_out_path(Path("files-from-zip")):
            if is_python_file(file_path):
                summary.write(f'File name: {file_path}\n')
            # we use one line, super convinient tree-hugger API call to get the needed data
                if pp.parse_file(str(file_path)):
                    temp_cache = []
                    # The following call returns a dict where each key is a name of a function
                    # And each value is a tuple, (function_body, function_docstring)
                    func_and_docstr = pp.get_all_function_bodies(strip_docstr=True)
                    for func_name, (body, docstr) in func_and_docstr.items():
                        example = [Example(source=body, target=None)]
                        message, length = inference(get_features(example, tokenizer), model, tokenizer)
                        summary.write(f'Summary: {message}\n\n')
    summary.close()
    return False
