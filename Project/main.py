#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import os
import sys
import time
import json
import random
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import set_seed

from dataset.DatasetLoader import DatasetLoader
from dataset.DatasetTorch import DatasetMultiChoiceQA
from model.ModelLoader import ModelLoader
from model.TokenizerLoader import TokenizerLoader


def load_dataset(ds_name: str, verbose: bool = True) -> dict:
    # Dataset
    datasetLoader = DatasetLoader()
    dataset_hf, test_label = datasetLoader.get_dataset(ds_name=ds_name)

    # Dataset split (training/validation/test sets)
    dataset_hf = dataset_hf.shuffle(seeds=RANDOM_SEED)
    ds_hf_train, ds_hf_valid = dataset_hf["train"], dataset_hf["validation"]
    if test_label:
        ds_hf_test = dataset_hf["test"]
    else:  # split half of the validation set as the test set
        dataset_split = ds_hf_valid.train_test_split(test_size=0.5, shuffle=False)
        ds_hf_valid = dataset_split["train"]
        ds_hf_test = dataset_split["test"]
        dataset_hf["validation"] = ds_hf_valid
        dataset_hf["test"] = ds_hf_test
    del dataset_split

    # Show dataset information
    if verbose:
        print(f"[Dataset] Training set shape: {ds_hf_train.shape}")
        print(f"[Dataset] Validation set shape: {ds_hf_valid.shape}")
        print(f"[Dataset] Test set shape: {ds_hf_test.shape}")
        assert ds_hf_train.column_names == ds_hf_valid.column_names == ds_hf_test.column_names, \
            "Assertion Error: column_names mismatch"
        print(f"[Dataset] column names: {ds_hf_train.column_names}")
        print(f"[Dataset] features: {ds_hf_train.features}\n")

    # Set in-context learning examples (random choice at least 3 examples from the training set)
    icl_indices = random.sample(range(len(ds_hf_train)), max(3, N_ICL))
    icl_dataset = ds_hf_train.select(icl_indices)
    icl_prompt = ""
    for icl_item in icl_dataset:
        icl_item = datasetLoader.map_prompt(icl_item)  # get the prompt (without answer)
        cur_prompt = icl_item["prompt"] + f"Answer: {icl_item['answer']}\n\n"  # set the answer for the ICL example
        icl_prompt += cur_prompt
    # icl_prompt_len = len(icl_prompt)
    if verbose:
        print(f"[Prompt] In-context Learning ({N_ICL} examples):\n{icl_prompt}")

    return {
        "dataset_hf": dataset_hf,
        "icl_prompt": icl_prompt,
    }


def load_model(model_name: str, verbose: bool = True) -> dict:
    # Model
    modelLoader = ModelLoader()
    model = modelLoader.get_model(model_name=model_name)
    model.to(DEVICE)
    # model.train()
    # model.eval()

    # Show model information
    if verbose:
        print(f"[Model] Parameters (total): {model.num_parameters()}")
        print(f"[Model] Parameters (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return {
        "model": model,
    }


def load_tokenizer(model_name: str, is_train: bool = True, verbose: bool = True) -> dict:
    # Tokenizer
    tokenizerLoader = TokenizerLoader()
    if is_train:
        tokenizer = tokenizerLoader.get_tokenizer(
            model_name=model_name, padding_side="right", truncation_side="right")
    else:
        tokenizer = tokenizerLoader.get_tokenizer(
            model_name=model_name, padding_side="left", truncation_side="left")

    # Special tokens
    # pad_token = "<|padoftext|>"
    # tokenizer.add_tokens([pad_token], special_tokens=True)
    # tokenizer.pad_token = pad_token
    # tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_tokens(["<bos>", "<eos>", "<unk>", "<pad>"], special_tokens=True)
    # tokenizer.bos_token = "<bos>"
    # tokenizer.eos_token = "<eos>"
    # tokenizer.unk_token = "<unk>"
    # tokenizer.pad_token = "<pad>"
    # tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    # tokenizer.add_special_tokens({"bos_token": "[BOS]"})

    # Show tokenizer information
    if verbose:
        print(f"[Tokenizer] (is_train: {is_train}) vocab size: {tokenizer.vocab_size}")
        print(f"[Tokenizer] (is_train: {is_train}) all special tokens: {tokenizer.all_special_tokens}\n")

    return {
        "tokenizer": tokenizer,
    }


def finetune(
        ds_name: str,
        model_name: str,
        ft_model,
        tokenizer,
        dataloader: DataLoader,
        icl_prompt: str = "",
        save_dir: str = "",
        verbose: bool = False,
):
    """
    Model fine-tuning. Save the model checkpoints and tokenizer during/after training.
    :param ds_name: the dataset name.
    :param model_name: the model name.
    :param ft_model: the model to fine-tune.
    :param tokenizer: the tokenizer for training (right padding).
    :param dataloader: the dataloader (mini-batched input/output).
    :param icl_prompt: the prefix prompt for in-context learning.
    :param save_dir: specified directory for saving the model checkpoints and logs.
    :param verbose: verbose model: print logs.
    :return: the tuned model and used tokenizer.
    """

    ft_model.train()

    if not isinstance(save_dir, str) or len(save_dir) == 0:
        save_dir = f"{ds_name}---{model_name}"

    # optimizer = optim.Adam(ft_model.parameters(), lr=float(1e-3), weight_decay=float(5e-4))
    optimizer = optim.Adam(ft_model.parameters(), lr=INIT_LR, weight_decay=W_DECAY)
    if verbose:
        print(optimizer)

    all_losses = []  # store the loss of each batch
    loss_logs = []
    batch_cnt = 0
    # show_gap = 1000
    show_gap = 100

    for epoch in range(EPOCH):
        if verbose:
            print(f"\n\n>>> Epoch: {epoch}")

        for batch_idx, batch_train in enumerate(dataloader):
            # Raw inputs
            answer_list = batch_train["answer"]
            prompt_list = batch_train["prompt"]
            prompt_list = [icl_prompt + prompt + "Answer: " for prompt in prompt_list]  # add ICL examples
            assert len(prompt_list) == len(answer_list), f"Assertion Error: len(prompt_list) != len(answer_list)"
            input_list = [prompt + answer for prompt, answer in zip(prompt_list, answer_list)]

            # Tokenized inputs
            inputs = tokenizer(input_list, return_tensors="pt", padding=True)  # padding_side="right"
            inputs.data["labels"] = inputs.data["input_ids"].clone()  # inputs.input_ids.clone()
            inputs = inputs.to(DEVICE)

            # Forward pass
            outputs = ft_model(**inputs, output_hidden_states=True, output_attentions=True)  # use_cache=True
            # type(outputs) is <class 'transformers.modeling_outputs.CausalLMOutputWithCrossAttentions'>
            # last_hidden_states = outputs.hidden_states[-1]  # outputs.last_hidden_state
            loss = outputs.loss  # Language modeling loss (for next-token prediction).

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss_value = loss.detach().cpu().numpy()
            loss_value = loss.detach().cpu().numpy().item()
            all_losses.append(loss_value)

            # Training log
            if batch_cnt % show_gap == 0:
                cur_log = f"[Total {batch_cnt + 1} batches]>>> Epoch {epoch} >>> Batch {batch_idx} >>> loss: {loss}\n"
                if verbose:
                    print(cur_log)
                loss_logs.append(cur_log)
            batch_cnt += 1

        # After each epoch, save the model checkpoint
        save_ckpt_dir = os.path.join(CKPT_DIR, save_dir)
        if not os.path.isdir(save_ckpt_dir):
            os.makedirs(save_ckpt_dir, exist_ok=True)
        # save_ckpt_path = os.path.join(save_ckpt_dir, f"{ds_name}---{model_name}---Epoch_{epoch}---ckpt.pt")
        save_ckpt_path = os.path.join(save_ckpt_dir, f"Epoch_{epoch}---ckpt.pt")
        torch.save(ft_model.state_dict(), save_ckpt_path)

    # After fine-tuning, save the loss logs
    save_log_dir = os.path.join(LOG_DIR, save_dir)
    if not os.path.isdir(save_log_dir):
        os.makedirs(save_log_dir, exist_ok=True)
    # save_loss_path = os.path.join(save_log_dir, f"{ds_name}---{model_name}---all_losses.log")
    save_loss_path = os.path.join(save_log_dir, f"all_losses.log")
    with open(save_loss_path, "w", encoding="utf-8") as fp_out:
        fp_out.writelines(all_losses)
    # save_log_path = os.path.join(save_log_dir, f"{ds_name}---{model_name}---loss_logs.log")
    save_log_path = os.path.join(save_log_dir, f"loss_logs.log")
    with open(save_log_path, "w", encoding="utf-8") as fp_out:
        fp_out.writelines(loss_logs)

    return {
        "ft_model": ft_model,
        "tokenizer": tokenizer,
        "all_losses": all_losses,
        "loss_logs": loss_logs,
    }


def generate(
        ds_name: str,
        model_name: str,
        gen_model,
        tokenizer,
        dataloader: DataLoader,
        icl_prompt: str = "",
        save_dir: str = "",
        save_fn: str = "",
        verbose: bool = False,
):
    """
        Model generation for evaluation
        :param ds_name: the dataset name.
        :param model_name: the model name.
        :param gen_model: the model for generation.
        :param tokenizer: the tokenizer for generation (left padding).
        :param dataloader: the dataloader (mini-batched input/output).
        :param icl_prompt: the prefix prompt for in-context learning.
        :param save_dir: specified directory for saving the results.
        :param save_fn: specified filename for saving the results.
        :param verbose: verbose model: print logs.
        :return: None. Save the evaluation results/scores after generation.
        """
    gen_model.eval()

    if not isinstance(save_dir, str) or len(save_dir) == 0:
        save_dir = f"{ds_name}---{model_name}"

    all_raw_preds = []
    all_preds = []
    all_answers = []
    icl_prompt_len = len(icl_prompt)

    for batch_idx, batch_train in enumerate(dataloader):
        # Raw inputs
        answer_list = batch_train["answer"]
        prompt_list = batch_train["prompt"]
        prompt_list = [icl_prompt + prompt + "Answer: " for prompt in prompt_list]  # add ICL examples
        assert len(prompt_list) == len(answer_list), f"Assertion Error: len(prompt_list) != len(answer_list)"
        # input_list = [prompt + answer for prompt, answer in zip(prompt_list, answer_list)]
        input_list = prompt_list

        # Tokenized inputs
        inputs = tokenizer(input_list, return_tensors="pt", padding=True)  # padding_side="right"
        # inputs.data["labels"] = inputs.data["input_ids"].clone()
        inputs = inputs.to(DEVICE)
        max_input_len = int(inputs.input_ids.shape[-1])

        # Forward pass
        # outputs = gen_model(**inputs, output_hidden_states=True, output_attentions=True)  # use_cache=True
        # loss = outputs.loss  # Language modeling loss (for next-token prediction).

        # Generate (beam search, sampling, temperature, etc.)
        with torch.no_grad():
            outputs = gen_model.generate(
                inputs.input_ids,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # max_length=512,
                max_length=max_input_len + LEN_GEN,
                num_beams=5,
                no_repeat_ngram_size=2,
                num_return_sequences=N_GEN,
                temperature=0.7,
                # temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True
            )

        # Decoding
        outputs_decode = tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw_preds = []
        preds = []
        for output_decode in outputs_decode:
            if isinstance(output_decode, str):
                raw_pred = output_decode
            elif isinstance(output_decode, list):
                raw_pred = output_decode[0]
            else:
                raise ValueError(f"ValueError: type(output_decode) = {type(output_decode)}")

            raw_pred = raw_pred[icl_prompt_len:]
            raw_preds.append(raw_pred)

            find_ans = re.findall(r"Answer:\s*(\S.*)", raw_pred)
            preds.append(find_ans[0])

        # Store prompts, outputs, and answers
        assert len(raw_preds) == len(preds) == len(answer_list)
        all_raw_preds += raw_preds
        all_preds += preds
        all_answers += answer_list

    results = []
    correct_idx = []
    incorrect_idx = []
    assert len(all_raw_preds) == len(all_preds) == len(all_answers) > 0
    for idx, (raw_pred, pred, answer) in enumerate(zip(all_raw_preds, all_preds, all_answers)):
        if pred.strip() == answer.strip():
            correct_idx.append(idx)
            cur_score = 1
        else:
            incorrect_idx.append(idx)
            cur_score = 0

        cur_result = {
            "prompt": raw_pred,
            "pred": pred,
            "answer": answer,
            "score": cur_score,
        }
        results.append(cur_result)

    accuracy = len(correct_idx) / len(results)
    if verbose:
        print(f">>> Accuracy: {accuracy:.2f}")  # GPT-2 on Commonsense QA: valid (before FT) 0.17540983606557378

    # Save the evaluation results
    save_results_dir = os.path.join(OUTPUT_DIR, save_dir)
    if not os.path.isdir(save_results_dir):
        os.makedirs(save_results_dir, exist_ok=True)
    if isinstance(save_fn, str) and len(save_fn) > 0:
        save_results_path = os.path.join(save_results_dir, save_fn)
    else:
        # save_results_path = os.path.join(save_results_dir, f"{ds_name}---{model_name}---results.jsonl")
        save_results_path = os.path.join(save_results_dir, f"results.jsonl")
    with open(save_results_path, "w", encoding="utf-8") as fp_out:
        for result in results:
            fp_out.write(json.dumps(result) + "\n")


def run(verbose: bool = False) -> None:
    # Dataset
    dataset_dict = load_dataset(ds_name=args.ds_name, verbose=verbose)
    dataset_hf = dataset_dict["dataset_hf"]
    icl_prompt = dataset_dict["icl_prompt"]

    # Model
    model_dict = load_model(model_name=args.model_name, verbose=verbose)
    model = model_dict["model"]

    # Tokenizer
    tokenizer_train_dict = load_tokenizer(model_name=args.model_name, is_train=True, verbose=verbose)
    tokenizer_train = tokenizer_train_dict["tokenizer"]
    tokenizer_eval_dict = load_tokenizer(model_name=args.model_name, is_train=False, verbose=verbose)
    tokenizer_eval = tokenizer_eval_dict["tokenizer"]

    # Convert Hugging Face datasets to PyTorch Dataset and DataLoader (mini-batch)
    ds_torch_train = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_train, splits="train")
    ds_torch_valid = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_eval, splits="validation")
    ds_torch_test = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_eval, splits="test")
    dataloader_train = DataLoader(ds_torch_train, batch_size=BSZ_TRAIN, shuffle=False)
    dataloader_valid = DataLoader(ds_torch_valid, batch_size=BSZ_GEN, shuffle=False)
    dataloader_test = DataLoader(ds_torch_test, batch_size=BSZ_GEN, shuffle=False)

    # Evaluation on the valid set before fine-tuning
    if verbose:
        print("\n\nEvaluation on the valid set before fine-tuning...")
    generate(
        ds_name=args.ds_name,
        model_name=args.model_name,
        gen_model=model,
        tokenizer=tokenizer_eval,
        dataloader=dataloader_valid,
        icl_prompt=icl_prompt,
        save_dir="",  # f"{ds_name}---{model_name}"
        save_fn=f"results_beforeFT_valid.jsonl",
    )

    # Evaluation on the test set before fine-tuning
    if verbose:
        print("\n\nEvaluation on the test set before fine-tuning...")
    generate(
        ds_name=args.ds_name,
        model_name=args.model_name,
        gen_model=model,
        tokenizer=tokenizer_eval,
        dataloader=dataloader_test,
        icl_prompt=icl_prompt,
        save_dir="",  # f"{ds_name}---{model_name}"
        save_fn=f"results_beforeFT_test.jsonl",
    )

    # Fine-tune the model (Causal LM, next token prediction)
    if verbose:
        print("\n\nFine-tune the model (Causal LM, next token prediction)...")
    ft_dict = finetune(
        ds_name=args.ds_name,
        model_name=args.model_name,
        ft_model=model,
        tokenizer=tokenizer_eval,
        dataloader=dataloader_train,
        icl_prompt=icl_prompt,
        save_dir="",  # f"{ds_name}---{model_name}"
    )
    ft_model = ft_dict["ft_model"]
    # tokenizer = ft_dict["tokenizer"]
    # all_losses = ft_dict["all_losses"]
    # loss_logs = ft_dict["loss_logs"]

    # Evaluation on the valid set after fine-tuning
    if verbose:
        print("\n\nEvaluation on the valid set after fine-tuning...")
    generate(
        ds_name=args.ds_name,
        model_name=args.model_name,
        gen_model=ft_model,
        tokenizer=tokenizer_eval,
        dataloader=dataloader_valid,
        icl_prompt=icl_prompt,
        save_dir="",  # f"{ds_name}---{model_name}"
        save_fn=f"results_afterFT_valid.jsonl",
    )

    # Evaluation on the test set after fine-tuning
    if verbose:
        print("\n\nEvaluation on the test set after fine-tuning...")
    generate(
        ds_name=args.ds_name,
        model_name=args.model_name,
        gen_model=ft_model,
        tokenizer=tokenizer_eval,
        dataloader=dataloader_test,
        icl_prompt=icl_prompt,
        save_dir="",  # f"{ds_name}---{model_name}"
        save_fn=f"results_afterFT_test.jsonl",
    )

    if verbose:
        print("\n\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose model: print logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device")
    parser.add_argument("-d", "--ds_name", type=str, default="", help="Dataset name, e.g., commonsense_qa")
    parser.add_argument("-m", "--model_name", type=str, default="", help="Model name (Causal LM), e.g., gpt2")
    parser.add_argument("--n_icl", type=int, default=5, help="The number of examples for in-context learning")
    parser.add_argument("--n_gen", type=int, default=1, help="The number of sentences to be generated")
    parser.add_argument("--len_gen", type=int, default=10, help="The number of max tokens to be generated")
    parser.add_argument("--epoch", type=int, default=5, help="The number of epochs for training")
    parser.add_argument("--bsz_train", type=int, default=32, help="The batch size for training")
    parser.add_argument("--bsz_gen", type=int, default=32, help="The batch size for generation / evaluation")
    parser.add_argument("--init_lr", type=float, default=float(1e-3), help="The initial learning rate for training")
    parser.add_argument("--w_decay", type=float, default=float(5e-4), help="The weight decay rate for training")
    parser.add_argument("--log_dir", type=str, default="log", help="The directory to save logs")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="The directory to save model checkpoints")
    parser.add_argument("--output_dir", type=str, default="output", help="The directory to outputs, e.g., results")
    args = parser.parse_args()
    print(args)

    timer_start = time.perf_counter()

    # Set the random seed of all modules
    RANDOM_SEED = int(args.seed)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    # Hyperparameters
    VERBOSE = bool(args.verbose)  # Verbose model: print logs
    EPOCH = int(args.epoch)  # The number of epochs for training
    BSZ_TRAIN = int(args.bsz_train)  # The batch size for training
    BSZ_GEN = int(args.bsz_gen)  # The batch size for generation /  evaluation
    INIT_LR = float(args.init_lr)  # The initial learning rate for training
    W_DECAY = float(args.w_decay)  # The weight decay rate for training
    N_ICL = int(args.n_icl)  # The number of examples for in-context learning
    N_GEN = int(args.n_gen)  # The number of sentences to be generated (for each generate(...) call)
    LEN_GEN = int(args.len_gen)  # The number of max tokens to be generated
    LOG_DIR = str(args.ckpt_dir)  # The directory to save logs
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    CKPT_DIR = str(args.ckpt_dir)  # The directory to save model checkpoints
    if not os.path.isdir(CKPT_DIR):
        os.makedirs(CKPT_DIR, exist_ok=True)
    OUTPUT_DIR = str(args.output_dir)  # The directory to outputs, e.g., results
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CUDA
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cpu" if not HAS_CUDA else "cuda")
    if VERBOSE:
        print(f"HAS_CUDA: {HAS_CUDA}; DEVICE: {DEVICE}")
        print("torch.__version__:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("torch.cuda.get_arch_list():", torch.cuda.get_arch_list())
        if HAS_CUDA:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
            print("torch.cuda.current_device():", torch.cuda.current_device())
            print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))

    run(verbose=VERBOSE)

    timer_end = time.perf_counter()
    print("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
