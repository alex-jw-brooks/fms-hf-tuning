# Copyright The IBM Tuning Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit Tests for SFT Trainer.
"""

# Standard
import json
import os
import tempfile

# First Party
from scripts.run_inference import TunedCausalLM
from tests.data import EMPTY_DATA, MALFORMATTED_DATA, TWITTER_COMPLAINTS_DATA
from tests.helpers import causal_lm_train_kwargs

# Third Party
from datasets.exceptions import DatasetGenerationError
import pytest
import torch
import transformers

# Local
from tuning import sft_trainer

HAPPY_PATH_KWARGS = {
    "model_name_or_path": "Maykeye/TinyLLama-v0",
    "data_path": TWITTER_COMPLAINTS_DATA,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.00001,
    "weight_decay": 0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "include_tokens_per_second": True,
    "packing": False,
    "response_template": "\n### Label:",
    "dataset_text_field": "output",
    "use_flash_attn": False,
    "torch_dtype": "float32",
    "model_max_length": 4096,
    "peft_method": "pt",
    "prompt_tuning_init": "RANDOM",
    "num_virtual_tokens": 8,
    "prompt_tuning_init_text": "hello",
    "tokenizer_name_or_path": "Maykeye/TinyLLama-v0",
    "save_strategy": "epoch",
}


def test_run_causallm_pt():
    """Check if we can bootstrap and run causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"output_dir": tempdir}}

        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir, "PROMPT_TUNING")

        # Load the tuned model
        loaded_model = TunedCausalLM.load(
            checkpoint_path=os.path.join(tempdir, "checkpoint-5"),
        )

        # Run inference on the text using the tuned model
        loaded_model.run(
            "Simply put, the theory of relativity states that ", max_new_tokens=500
        )


def test_run_causallm_lora():
    """Check if we can bootstrap and run causallm models"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"output_dir": tempdir, "peft_method": "lora"}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir, "LORA")

        # Load the tuned model
        loaded_model = TunedCausalLM.load(
            checkpoint_path=os.path.join(tempdir, "checkpoint-5"),
        )

        # Run inference on the text using the tuned model
        loaded_model.run(
            "Simply put, the theory of relativity states that ", max_new_tokens=500
        )


def _validate_training(tempdir, peft_type):
    assert any(x.startswith("checkpoint-") for x in os.listdir(tempdir))
    train_loss_file_path = "{}/train_loss.jsonl".format(tempdir)
    assert os.path.exists(train_loss_file_path) == True
    assert os.path.getsize(train_loss_file_path) > 0
    adapter_config_path = os.path.join(tempdir, "checkpoint-1", "adapter_config.json")
    assert os.path.exists(adapter_config_path)
    with open(adapter_config_path) as f:
        data = json.load(f)
        assert data.get("peft_type") == peft_type


### Tests for a variety of edge cases and potentially problematic cases;
# some of these test directly test validation within external dependencies
# and validate errors that we expect to get from them which might be unintuitive.
# In such cases, it would probably be best for us to handle these things directly
# for better error messages, etc.

### Tests related to tokenizer configuration
def test_tokenizer_has_no_eos_token():
    """Ensure that if the model has no EOS token, it sets the default before formatting."""
    # This is a bit roundabout, but patch the tokenizer and export it and the model to a tempdir
    # that we can then reload out of for the train call, and clean up afterwards.
    tokenizer = transformers.AutoTokenizer.from_pretrained(HAPPY_PATH_KWARGS["model_name_or_path"])
    model = transformers.AutoModelForCausalLM.from_pretrained(HAPPY_PATH_KWARGS["model_name_or_path"])
    tokenizer.eos_token = None
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer.save_pretrained(tempdir)
        model.save_pretrained(tempdir)
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"model_name_or_path": tempdir, "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        # If we handled this badly, we would probably get something like a
        # TypeError: can only concatenate str (not "NoneType") to str error
        # when we go to apply the data formatter.
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir, "PROMPT_TUNING")


### Tests for Bad dataset specification, i.e., data is valid, but the field we point it at isn't
def test_invalid_dataset_text_field():
    """Ensure that if we specify a dataset_text_field that doesn't exist, we get a KeyError."""
    TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"dataset_text_field": "not found", "output_dir": "foo/bar/baz"}}
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(KeyError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)


### Tests for bad training data (i.e., data_path is an unhappy value or points to an unhappy thing)
def test_malformatted_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"data_path": MALFORMATTED_DATA, "output_dir": "foo/bar/baz"}}
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(DatasetGenerationError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)

def test_empty_data():
    """Ensure that malformatted data explodes due to failure to generate the dataset."""
    TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"data_path": EMPTY_DATA, "output_dir": "foo/bar/baz"}}
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(DatasetGenerationError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)

def test_data_path_does_not_exist():
    """Ensure that we get a FileNotFoundError if the data is missing completely."""
    TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"data_path": "/foo/bar/foobar", "output_dir": "foo/bar/baz"}}
    model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
        TRAIN_KWARGS
    )
    with pytest.raises(FileNotFoundError):
        sft_trainer.train(model_args, data_args, training_args, tune_config)

def test_data_path_is_a_directory():
    """Ensure that we get FileNotFoundError if we point the data path at a dir, not a file."""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"data_path": tempdir, "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        # Confusingly, if we pass a directory for our data path, it will throw a FileNotFoundError saying
        # "unable to find '<data_path>'", since it can't find a matchable file in the path.
        with pytest.raises(FileNotFoundError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


### Tests for bad tuning module configurations
def test_run_causallm_lora_with_invalid_modules():
    """Check that we throw a value error if the target modules for lora don't exist."""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"peft_method": "lora", "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        # Defaults are q_proj / v_proj; this will fail lora as the torch module doesn't have them
        tune_config.target_modules = ["foo", "bar"]
        # Peft should throw a value error about modules not matching the base module
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)

### Direct validation tests based on whether or not packing is enabled
def test_no_packing_needs_dataset_text_field():
    """Ensure we need to set the dataset text field if packing is False"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"dataset_text_field": None, "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)

# TODO: Fix this case
@pytest.mark.skip(reason="currently crashes before validation is done")
def test_no_packing_needs_reponse_template():
    """Ensure we need to set the response template if packing is False"""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"response_template": None, "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)


### Tests for model dtype edge cases
@pytest.mark.skipif(torch.cuda.is_bf16_supported(), reason="Only runs if bf16 is unsupported")
def test_bf16_still_tunes_if_unsupported():
    """Ensure that even if bf16 is not supported, tuning still works without problems."""
    assert not torch.cuda.is_bf16_supported()
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"torch_dtype": "bfloat16", "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        sft_trainer.train(model_args, data_args, training_args, tune_config)
        _validate_training(tempdir, "PROMPT_TUNING")

def bad_torch_dtype():
    """Ensure that specifying an invalid torch dtype yields a ValueError."""
    with tempfile.TemporaryDirectory() as tempdir:
        TRAIN_KWARGS = {**HAPPY_PATH_KWARGS, **{"torch_dtype": "not a type", "output_dir": tempdir}}
        model_args, data_args, training_args, tune_config = causal_lm_train_kwargs(
            TRAIN_KWARGS
        )
        with pytest.raises(ValueError):
            sft_trainer.train(model_args, data_args, training_args, tune_config)
