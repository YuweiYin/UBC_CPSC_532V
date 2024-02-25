#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import os
import sys
import time
import json
import random
import logging
import argparse

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import set_seed

from dataset.DatasetLoader import DatasetLoader
from dataset.DatasetTorch import DatasetMultiChoiceQA
from model.ModelLoader import ModelLoader
from model.TokenizerLoader import TokenizerLoader


def training(
        ds_name: str,
        model_name: str,
        ft_model,
        tokenizer_train,
        tokenizer_eval,
        dataloader_train: DataLoader,
        dataloader_valid: DataLoader = None,
        dataloader_test: DataLoader = None,
        icl_prompt: str = "",
        do_eval_epoch: bool = True,
        do_eval_batch: bool = False,
        save_after_epoch: bool = True,
        eval_gap: int = 1000,
        logging_gap: int = 100,
        save_dir: str = "",
        verbose: bool = False,
):
    """
    Model training. Save the model checkpoints and tokenizer during/after training.
    :param ds_name: the dataset name.
    :param model_name: the model name.
    :param ft_model: the model to train.
    :param tokenizer_train: the tokenizer for training (right padding).
    :param tokenizer_eval: the tokenizer for generation (left padding).
    :param dataloader_train: the dataloader of the training set (mini-batched input/output).
    :param dataloader_valid: the dataloader of the valid set (mini-batched input/output) to pick the best model.
    :param dataloader_test: the dataloader of the test set (mini-batched input/output) if possible.
    :param icl_prompt: the prefix prompt for in-context learning.
    :param do_eval_epoch: whether run evaluation after each epoch or not.
    :param do_eval_batch: whether run evaluation per `eval_gap` batches or not. (If so, we will save the best model.)
    :param save_after_epoch: whether save the ckpt and results after each epoch.
    :param eval_gap: run evaluation per `eval_gap` batches.
    :param logging_gap: show loss per `logging_gap` batches.
    :param save_dir: specified directory for saving the model checkpoints and logs.
    :param verbose: verbose model: print logs.
    :return: the tuned model and used tokenizer.
    """

    ft_model.train()
    ft_model = ft_model.to(DEVICE)

    if not isinstance(save_dir, str) or len(save_dir) == 0:
        save_dir = f"{ds_name}---{model_name}"

    save_ckpt_dir = os.path.join(CKPT_DIR, save_dir)
    if not os.path.isdir(save_ckpt_dir):
        os.makedirs(save_ckpt_dir, exist_ok=True)

    save_log_dir = os.path.join(LOG_DIR, save_dir)
    if not os.path.isdir(save_log_dir):
        os.makedirs(save_log_dir, exist_ok=True)

    # optimizer = optim.Adam(ft_model.parameters(), lr=float(1e-3), weight_decay=float(5e-4))
    optimizer = optim.Adam(ft_model.parameters(), lr=INIT_LR, weight_decay=W_DECAY)
    if verbose:
        logger.info(optimizer)

    all_losses = []  # store the loss of each batch (divided by epochs)
    # loss_logs = []
    all_valid_scores = []  # store the accuracy score on the valid set after evaluation (divided by epochs)
    all_test_scores = []  # store the accuracy score on the test set after evaluation (divided by epochs)
    best_valid_score = 0.0
    best_test_score = 0.0

    batch_cnt = 0
    for epoch in range(EPOCH):
        if verbose:
            logger.info(f"\n\n>>> Epoch: {epoch}")
        epoch_losses = []

        for batch_idx, batch_train in enumerate(dataloader_train):
            # Raw inputs
            answer_list = batch_train["answer"]
            prompt_list = batch_train["prompt"]
            # prompt_list = [icl_prompt + prompt + "Answer: " for prompt in prompt_list]  # add ICL examples
            prompt_list = [prompt + "Answer: " for prompt in prompt_list]  # without ICL examples
            assert len(prompt_list) == len(answer_list), f"Assertion Error: len(prompt_list) != len(answer_list)"
            input_list = [prompt + answer for prompt, answer in zip(prompt_list, answer_list)]

            # Tokenized inputs
            inputs = tokenizer_train(input_list, return_tensors="pt", padding=True)  # padding_side="right"
            inputs.data["labels"] = inputs.data["input_ids"].clone()  # inputs.input_ids.clone()
            inputs = inputs.to(DEVICE)

            # Forward pass
            outputs = ft_model(**inputs, output_hidden_states=True, output_attentions=True)  # use_cache=True
            # type(outputs) is <class 'transformers.modeling_outputs.CausalLMOutputWithCrossAttentions'>
            # last_hidden_states = outputs.hidden_states[-1]  # outputs.last_hidden_state
            loss = outputs.loss  # Language modeling loss (for next-token prediction).

            # Backpropagation
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            # loss_value = loss.detach().cpu().numpy().item()
            loss_value = loss.sum().detach().cpu().numpy().item()
            epoch_losses.append(loss_value)

            # Training log
            if batch_cnt % logging_gap == 0:
                cur_log = f"[LOG] >>> Epoch {epoch} >>> Total Batch {batch_cnt + 1} >>> loss: {loss}"
                if verbose:
                    logger.info(cur_log)
                # loss_logs.append(cur_log)

            # Run evaluation
            if do_eval_batch and batch_cnt % eval_gap == 0:
                if isinstance(dataloader_valid, DataLoader):
                    # Evaluation on the valid set during training
                    if verbose:
                        logger.info(f"\n[Evaluation - valid] >>> Epoch {epoch} >>> Total Batch {batch_cnt + 1} "
                                    f">>> loss: {loss}")
                    valid_score = evaluate(
                        ds_name=ds_name,
                        model_name=model_name,
                        gen_model=ft_model,
                        tokenizer_eval=tokenizer_eval,
                        dataloader=dataloader_valid,
                        icl_prompt=icl_prompt,
                        save_dir=save_dir,  # f"{ds_name}---{model_name}"
                        save_fn=f"results---valid---batch{batch_cnt + 1}_epoch{epoch}.jsonl",
                        verbose=verbose,
                    )
                    all_valid_scores.append(valid_score)
                    if valid_score > best_valid_score:
                        best_valid_score = valid_score

                        # Save the best model based on the best valid score
                        save_ckpt_hf = os.path.join(save_ckpt_dir, "model_best")  # HF model folder
                        if not os.path.isdir(save_ckpt_hf):
                            os.makedirs(save_ckpt_hf, exist_ok=True)
                        if verbose:
                            logger.info(f"Save the best model (valid score {best_valid_score}) at {save_ckpt_hf}")
                        ft_model.save_pretrained(save_ckpt_hf)  # config.json, generation_config.json, model.safetensors
                        # save_ckpt_torch_fp = os.path.join(save_ckpt_hf, f"torch_ckpt.pt")  # torch.save
                        # torch.save(ft_model.state_dict(), save_ckpt_torch_fp)  # duplicated

                        # Save the tokenizers
                        save_tokenizer_train_hf = os.path.join(save_ckpt_hf, "tokenizer_train")
                        if not os.path.isdir(save_tokenizer_train_hf):
                            os.makedirs(save_tokenizer_train_hf, exist_ok=True)
                        tokenizer_train.save_pretrained(save_tokenizer_train_hf)

                        save_tokenizer_eval_hf = os.path.join(save_ckpt_hf, "tokenizer_eval")
                        if not os.path.isdir(save_tokenizer_eval_hf):
                            os.makedirs(save_tokenizer_eval_hf, exist_ok=True)
                        tokenizer_train.save_pretrained(save_tokenizer_eval_hf)

                if isinstance(dataloader_test, DataLoader):
                    # Evaluation on the test set during training
                    if verbose:
                        logger.info(f"\n[Evaluation - test] >>> Epoch {epoch} >>> Total Batch {batch_cnt + 1} "
                                    f">>> loss: {loss}")
                    test_score = evaluate(
                        ds_name=ds_name,
                        model_name=model_name,
                        gen_model=ft_model,
                        tokenizer_eval=tokenizer_eval,
                        dataloader=dataloader_test,
                        icl_prompt=icl_prompt,
                        save_dir=save_dir,  # f"{ds_name}---{model_name}"
                        save_fn=f"results---test---batch{batch_cnt + 1}_epoch{epoch}.jsonl",
                        verbose=verbose,
                    )
                    all_test_scores.append(test_score)
                    if test_score > best_test_score:
                        best_test_score = test_score

            batch_cnt += 1

        # After each epoch, save the losses and scores
        if verbose:
            logger.info(f"\n\n[END of Epoch {epoch}]")
        all_losses.append(epoch_losses)
        avg_ep_loss = np.mean(epoch_losses)
        save_loss_path = os.path.join(save_log_dir, f"all_losses.log")
        with open(save_loss_path, "w", encoding="utf-8") as fp_out:
            for epoch_losses in all_losses:
                fp_out.write(json.dumps(epoch_losses) + "\n")

        save_valid_score_path = os.path.join(save_log_dir, f"all_valid_scores.log")
        with open(save_valid_score_path, "w", encoding="utf-8") as fp_out:
            for valid_score in all_valid_scores:
                fp_out.write(json.dumps(valid_score) + "\n")

        save_test_score_path = os.path.join(save_log_dir, f"all_test_scores.log")
        with open(save_test_score_path, "w", encoding="utf-8") as fp_out:
            for test_score in all_test_scores:
                fp_out.write(json.dumps(test_score) + "\n")

        # Save the model with tokenizers at the end of this epoch
        if save_after_epoch:
            save_ckpt_hf = os.path.join(save_ckpt_dir, f"model_epoch{epoch}")  # HF model folder
            if not os.path.isdir(save_ckpt_hf):
                os.makedirs(save_ckpt_hf, exist_ok=True)
            if verbose:
                logger.info(f"Save the model for epoch {epoch} at {save_ckpt_hf}")
            ft_model.save_pretrained(save_ckpt_hf)  # config.json, generation_config.json, model.safetensors
            # save_ckpt_torch_fp = os.path.join(save_ckpt_hf, f"torch_ckpt.pt")  # torch.save
            # torch.save(ft_model.state_dict(), save_ckpt_torch_fp)  # duplicated

            # Save the tokenizers
            save_tokenizer_train_hf = os.path.join(save_ckpt_hf, "tokenizer_train")
            if not os.path.isdir(save_tokenizer_train_hf):
                os.makedirs(save_tokenizer_train_hf, exist_ok=True)
            tokenizer_train.save_pretrained(save_tokenizer_train_hf)

            save_tokenizer_eval_hf = os.path.join(save_ckpt_hf, "tokenizer_eval")
            if not os.path.isdir(save_tokenizer_eval_hf):
                os.makedirs(save_tokenizer_eval_hf, exist_ok=True)
            tokenizer_train.save_pretrained(save_tokenizer_eval_hf)

        # Run evaluation
        if do_eval_epoch:
            if isinstance(dataloader_valid, DataLoader):
                # Evaluation on the valid set at the end of this epoch
                if verbose:
                    logger.info(f"\n[Evaluation - valid] >>> Epoch {epoch} >>> Total Batch {batch_cnt} "
                                f">>> average loss: {avg_ep_loss}")
                valid_score = evaluate(
                    ds_name=ds_name,
                    model_name=model_name,
                    gen_model=ft_model,
                    tokenizer_eval=tokenizer_eval,
                    dataloader=dataloader_valid,
                    icl_prompt=icl_prompt,
                    save_dir=save_dir,  # f"{ds_name}---{model_name}"
                    save_fn=f"results---valid---batch{batch_cnt}_epoch{epoch}_END.jsonl",
                    verbose=verbose,
                )
                all_valid_scores.append(valid_score)
                if valid_score > best_valid_score:
                    best_valid_score = valid_score

                    # Save the best model based on the best valid score
                    save_ckpt_hf = os.path.join(save_ckpt_dir, "model_best")  # HF model folder
                    if not os.path.isdir(save_ckpt_hf):
                        os.makedirs(save_ckpt_hf, exist_ok=True)
                    if verbose:
                        logger.info(f"Save the best model (valid score {best_valid_score}) at {save_ckpt_hf}")
                    ft_model.save_pretrained(save_ckpt_hf)  # config.json, generation_config.json, model.safetensors
                    # save_ckpt_torch_fp = os.path.join(save_ckpt_hf, f"torch_ckpt.pt")  # torch.save
                    # torch.save(ft_model.state_dict(), save_ckpt_torch_fp)  # duplicated

                    # Save the tokenizers
                    save_tokenizer_train_hf = os.path.join(save_ckpt_hf, "tokenizer_train")
                    if not os.path.isdir(save_tokenizer_train_hf):
                        os.makedirs(save_tokenizer_train_hf, exist_ok=True)
                    tokenizer_train.save_pretrained(save_tokenizer_train_hf)

                    save_tokenizer_eval_hf = os.path.join(save_ckpt_hf, "tokenizer_eval")
                    if not os.path.isdir(save_tokenizer_eval_hf):
                        os.makedirs(save_tokenizer_eval_hf, exist_ok=True)
                    tokenizer_train.save_pretrained(save_tokenizer_eval_hf)

            if isinstance(dataloader_test, DataLoader):
                # Evaluation on the test set at the end of this epoch
                if verbose:
                    logger.info(f"\n[Evaluation - test] >>> Epoch {epoch} >>> Total Batch {batch_cnt} "
                                f">>> average loss: {avg_ep_loss}")
                test_score = evaluate(
                    ds_name=ds_name,
                    model_name=model_name,
                    gen_model=ft_model,
                    tokenizer_eval=tokenizer_eval,
                    dataloader=dataloader_test,
                    icl_prompt=icl_prompt,
                    save_dir=save_dir,  # f"{ds_name}---{model_name}"
                    save_fn=f"results---test---batch{batch_cnt}_epoch{epoch}_END.jsonl",
                    verbose=verbose,
                )
                all_test_scores.append(test_score)
                if test_score > best_test_score:
                    best_test_score = test_score

    return {
        "ft_model": ft_model,
        "tokenizer_train": tokenizer_train,
        "tokenizer_eval": tokenizer_eval,
        "all_losses": all_losses,
        # "loss_logs": loss_logs,
        "all_valid_scores": all_valid_scores,
        "all_test_scores": all_test_scores,
        "best_valid_score": best_valid_score,
        "best_test_score": best_test_score,
    }


def evaluate(
        ds_name: str,
        model_name: str,
        gen_model,
        tokenizer_eval,
        dataloader: DataLoader,
        icl_prompt: str = "",
        do_forward: bool = True,
        choice_prob: bool = True,
        save_dir: str = "",
        save_fn: str = "",
        verbose: bool = False,
) -> float:
    """
    Model generation for evaluation
    :param ds_name: the dataset name.
    :param model_name: the model name.
    :param gen_model: the model for generation.
    :param tokenizer_eval: the tokenizer for generation (left padding).
    :param dataloader: the dataloader for evaluation (mini-batched input/output).
    :param icl_prompt: the prefix prompt for in-context learning.
    :param do_forward: use forward pass to generate one next token for prediction. Otherwise, beam search OR sampling.
    :param choice_prob: if `do_forward`, whether to only consider the probabilities of the choices, like "A" "B" "C".
    :param save_dir: specified directory for saving the results.
    :param save_fn: specified filename for saving the results.
    :param verbose: verbose model: print logs.
    :return: Accuracy score. Save the evaluation results/scores after generation.
    """
    gen_model.eval()
    gen_model = gen_model.to(DEVICE)

    if not isinstance(save_dir, str) or len(save_dir) == 0:
        save_dir = f"{ds_name}---{model_name}"

    all_prompts = []
    all_preds = []
    all_answers = []

    # Two metrics when using Causal LM for classification
    # 1. direct generation: consider all possible tokens in the vocab
    # 2. (more reasonable) only consider the probabilities of the choices, like "A" "B" "C"
    choice_ids = []
    choice_labels = []

    for batch_idx, batch_train in enumerate(dataloader):
        # Raw inputs
        answer_list = batch_train["answer"]
        prompt_list = batch_train["prompt"]
        # prompt_list = [icl_prompt + prompt + "Answer: " for prompt in prompt_list]  # add ICL examples
        # prompt_list = [prompt + "Answer: " for prompt in prompt_list]  # without ICL examples
        prompt_list = [prompt + "Answer:" for prompt in prompt_list]  # without ICL examples nor trailing space
        all_prompts += prompt_list

        assert len(prompt_list) == len(answer_list), f"Assertion Error: len(prompt_list) != len(answer_list)"
        # input_list = [prompt + answer for prompt, answer in zip(prompt_list, answer_list)]
        input_list = prompt_list

        # Get the tokenized ids of each choice, e.g., "A"/" A", "B"/" B", "C"/" C", etc.
        if len(choice_ids) == 0:
            c_labels = [_labels[0] for _labels in batch_train["choices_label"]]
            c_labels = list(set(c_labels))  # remove duplication
            c_labels.sort()
            for c_label in c_labels:  # get the vocab id
                c_label = c_label.strip()  # e.g., "A" or "B"
                c_label_ids = tokenizer_eval.encode(c_label)
                if len(c_label_ids) == 1:
                    choice_ids.append(c_label_ids[0])
                    choice_labels.append(c_label)
                c_label = " " + c_label  # e.g., " A" or " B"
                c_label_ids = tokenizer_eval.encode(c_label)
                if len(c_label_ids) == 1:
                    choice_ids.append(c_label_ids[0])
                    choice_labels.append(c_label)

        # Tokenized inputs
        inputs = tokenizer_eval(input_list, return_tensors="pt", padding=True)  # padding_side="right"
        # inputs.data["labels"] = inputs.data["input_ids"].clone()
        inputs = inputs.to(DEVICE)

        max_input_len = int(inputs.input_ids.shape[-1])
        input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]
        # seq_lengths = attention_mask.sum(-1)
        # input_decode = tokenizer_eval.batch_decode(
        #     input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if do_forward:
            # Forward pass
            with torch.no_grad():
                outputs = gen_model(**inputs, output_hidden_states=True, output_attentions=True)  # use_cache=True
            # loss = outputs.loss  # Language modeling loss (for next-token prediction).
            logits = outputs.logits
            if choice_prob:  # only consider the probabilities of the choices, like "A" "B" "C"
                pred_logits = logits[:, -1, choice_ids]
                pred_ids = pred_logits.argmax(-1)
                preds = [choice_labels[int(p)] for p in pred_ids]
            else:  # consider all possible tokens in the vocab
                pred_logits = logits[:, -1, :]
                pred_ids = pred_logits.argmax(-1)
                preds = tokenizer_eval.batch_decode(
                    pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        else:
            # Generate (beam search, sampling, temperature, etc.)
            with torch.no_grad():
                outputs_gen = gen_model.generate(
                    input_ids,
                    pad_token_id=tokenizer_eval.pad_token_id,
                    bos_token_id=tokenizer_eval.bos_token_id,
                    eos_token_id=tokenizer_eval.eos_token_id,
                    # max_length=512,
                    max_length=max_input_len + LEN_GEN,
                    num_beams=5,
                    repetition_penalty=1.1,
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
            outputs_decode = tokenizer_eval.batch_decode(
                outputs_gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # Find the prediction
            preds = []
            for output_decode in outputs_decode:
                if isinstance(output_decode, str):
                    raw_pred = output_decode
                elif isinstance(output_decode, list):
                    raw_pred = output_decode[0]
                else:
                    raise ValueError(f"ValueError: type(output_decode) = {type(output_decode)}")

                find_ans = re.findall(r"Answer:\s*(\S.*)", raw_pred)
                if len(find_ans) > 0:
                    preds.append(find_ans[0])
                else:
                    preds.append("NULL")

        # Store prompts, outputs, and answers
        assert len(preds) == len(answer_list)
        all_preds += preds
        all_answers += answer_list

    results = []
    correct_idx = []
    incorrect_idx = []
    assert len(all_prompts) == len(all_preds) == len(all_answers) > 0
    for idx, (prompt, pred, answer) in enumerate(zip(all_prompts, all_preds, all_answers)):
        if pred.strip() == answer.strip():
            correct_idx.append(idx)
            cur_score = 1
        else:
            incorrect_idx.append(idx)
            cur_score = 0

        cur_result = {
            "prompt": prompt,
            "pred": pred,
            "answer": answer,
            "score": cur_score,
        }
        results.append(cur_result)

    accuracy = len(correct_idx) / len(results) if len(results) > 0 else 0.0
    if verbose:
        # logger.info(f">>> Accuracy: {accuracy:.5f}")  # GPT-2 on Commonsense QA: (before FT) valid 0.18; test 0.16
        logger.info(f">>> Accuracy: {accuracy}")  # GPT-2 on Commonsense QA: valid (before FT) 0.17540983606557378

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

    return accuracy


def run(verbose: bool = False) -> None:
    # Loaders
    datasetLoader = DatasetLoader()
    modelLoader = ModelLoader()
    tokenizerLoader = TokenizerLoader()

    # Set the random seed of all modules (again)
    set_seed(RANDOM_SEED)

    # Dataset
    dataset_dict = datasetLoader.load_dataset(
        ds_name=args.ds_name, n_icl=N_ICL, cache_dir=CACHE_DIR, random_seed=RANDOM_SEED, verbose=verbose)
    dataset_hf = dataset_dict["dataset_hf"]
    icl_prompt = dataset_dict["icl_prompt"]

    # Model
    model_dict = modelLoader.load_model(model_name=args.model_name, cache_dir=CACHE_DIR, verbose=verbose)
    model = model_dict["model"]
    model.to(DEVICE)

    # Tokenizer
    tokenizer_train_dict = tokenizerLoader.load_tokenizer(
        model_name=args.model_name, is_train=True, cache_dir=CACHE_DIR, verbose=verbose)
    tokenizer_train = tokenizer_train_dict["tokenizer"]
    tokenizer_eval_dict = tokenizerLoader.load_tokenizer(
        model_name=args.model_name, is_train=False, cache_dir=CACHE_DIR, verbose=verbose)
    tokenizer_eval = tokenizer_eval_dict["tokenizer"]

    # Convert Hugging Face datasets to PyTorch Dataset and DataLoader (mini-batch)
    ds_torch_train = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_train, splits="train")
    ds_torch_valid = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_eval, splits="validation")
    ds_torch_test = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_eval, splits="test")
    dataloader_train = DataLoader(ds_torch_train, batch_size=BSZ_TRAIN, shuffle=False)
    dataloader_valid = DataLoader(ds_torch_valid, batch_size=BSZ_GEN, shuffle=False)
    dataloader_test = DataLoader(ds_torch_test, batch_size=BSZ_GEN, shuffle=False)

    if EVAL_BEFORE:
        # Evaluation on the valid set before training
        if verbose:
            logger.info("\n\nEvaluation on the valid set before training...")
        evaluate(
            ds_name=args.ds_name,
            model_name=args.model_name,
            gen_model=model,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_valid,
            icl_prompt=icl_prompt,
            save_dir=SAVE_DIR,  # f"{ds_name}---{model_name}"
            save_fn=f"results_beforeFT_valid.jsonl",
            verbose=verbose,
        )
        # Evaluation on the test set before training
        if verbose:
            logger.info("\n\nEvaluation on the test set before training...")
        evaluate(
            ds_name=args.ds_name,
            model_name=args.model_name,
            gen_model=model,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_test,
            icl_prompt=icl_prompt,
            save_dir=SAVE_DIR,  # f"{ds_name}---{model_name}"
            save_fn=f"results_beforeFT_test.jsonl",
            verbose=verbose,
        )

    # Train (fine-tune) the model (Causal LM, next token prediction)
    if verbose:
        logger.info("\n\nTrain (fine-tune) the model (Causal LM, next token prediction)...")
    ft_dict = training(
        ds_name=args.ds_name,
        model_name=args.model_name,
        ft_model=model,
        tokenizer_train=tokenizer_train,
        tokenizer_eval=tokenizer_eval,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        dataloader_test=dataloader_test,
        icl_prompt=icl_prompt,
        # do_eval_epoch=True,
        # do_eval_batch=True,
        # save_after_epoch=True,
        # eval_gap=1000,
        # logging_gap=100,
        do_eval_epoch=DO_EVAL_EPOCH,
        do_eval_batch=DO_EVAL_BATCH,
        save_after_epoch=SAVE_AFTER_EPOCH,
        eval_gap=EVAL_GAP,
        logging_gap=LOGGING_GAP,
        save_dir=SAVE_DIR,  # f"{ds_name}---{model_name}"
        verbose=verbose,
    )
    ft_model = ft_dict["ft_model"]  # the trained model
    # tokenizer_train = ft_dict["tokenizer_train"]
    # tokenizer_eval = ft_dict["tokenizer_eval"]
    # all_losses = ft_dict["all_losses"]
    # # loss_logs = ft_dict["loss_logs"]
    # all_valid_scores = ft_dict["all_valid_scores"]
    # all_test_scores = ft_dict["all_test_scores"]
    # best_valid_score = ft_dict["best_valid_score"]
    # best_test_score = ft_dict["best_test_score"]

    if EVAL_AFTER:
        # Evaluation on the valid set after training
        if verbose:
            logger.info("\n\nEvaluation on the valid set after training...")
        evaluate(
            ds_name=args.ds_name,
            model_name=args.model_name,
            gen_model=ft_model,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_valid,
            icl_prompt=icl_prompt,
            save_dir=SAVE_DIR,  # f"{ds_name}---{model_name}"
            save_fn=f"results_afterFT_valid.jsonl",
            verbose=verbose,
        )
        # Evaluation on the test set after training
        if verbose:
            logger.info("\n\nEvaluation on the test set after training...")
        evaluate(
            ds_name=args.ds_name,
            model_name=args.model_name,
            gen_model=ft_model,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_test,
            icl_prompt=icl_prompt,
            save_dir=SAVE_DIR,  # f"{ds_name}---{model_name}"
            save_fn=f"results_afterFT_test.jsonl",
            verbose=verbose,
        )

    if verbose:
        logger.info("\n\nDone!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose model: print logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device(s), e.g., 0 OR 0,1")
    parser.add_argument("-d", "--ds_name", type=str, default="", help="Dataset name, e.g., commonsense_qa")
    parser.add_argument("-m", "--model_name", type=str, default="", help="Model name (Causal LM), e.g., gpt2")
    parser.add_argument("--eval_before", action="store_true", default=False, help="Run evaluation before training")
    parser.add_argument("--eval_after", action="store_true", default=False, help="Run evaluation after training")
    parser.add_argument("--do_eval_epoch", action="store_true", default=False,
                        help="Whether run evaluation after each epoch or not.")
    parser.add_argument("--do_eval_batch", action="store_true", default=False,
                        help="Whether run evaluation per `eval_gap` batches or not. (If so, save the best model.)")
    parser.add_argument("--save_after_epoch", action="store_true", default=False,
                        help="Whether save the ckpt and results after each epoch.")
    parser.add_argument("--eval_gap", type=int, default=1000, help="Run evaluation per `eval_gap` batches")
    parser.add_argument("--logging_gap", type=int, default=100, help="Show loss per `logging_gap` batches.")
    parser.add_argument("--n_icl", type=int, default=5, help="The number of examples for in-context learning")
    parser.add_argument("--n_gen", type=int, default=1, help="The number of sentences to be generated")
    parser.add_argument("--len_gen", type=int, default=10, help="The number of max tokens to be generated")
    parser.add_argument("--epoch", type=int, default=5, help="The number of epochs for training")
    parser.add_argument("--bsz_train", type=int, default=32, help="The batch size for training")
    parser.add_argument("--bsz_gen", type=int, default=32, help="The batch size for generation / evaluation")
    parser.add_argument("--init_lr", type=float, default=float(1e-3), help="The initial learning rate for training")
    parser.add_argument("--w_decay", type=float, default=float(5e-4), help="The weight decay rate for training")
    parser.add_argument("--save_dir", type=str, default="", help="The directory of the current run")
    parser.add_argument("--cache_dir", type=str, default="~/.cache/huggingface/",
                        help="The directory where data & model are cached")
    parser.add_argument("--log_dir", type=str, default="log", help="The directory to save logs")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="The directory to save model checkpoints")
    parser.add_argument("--output_dir", type=str, default="output", help="The directory to outputs, e.g., results")
    args = parser.parse_args()
    logger.info(args)

    timer_start = time.perf_counter()

    # Set the random seed of all modules
    RANDOM_SEED = int(args.seed)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    # Hyperparameters
    CUDA = str(args.cuda).strip()  # CUDA device(s), e.g., "0" OR "0,1"
    VERBOSE = bool(args.verbose)  # Verbose model: print logs
    EVAL_BEFORE = bool(args.eval_before)  # Run evaluation before training
    EVAL_AFTER = bool(args.eval_after)  # Run evaluation after training
    DO_EVAL_EPOCH = bool(args.do_eval_epoch)  # Run evaluation after each epoch
    DO_EVAL_BATCH = bool(args.do_eval_batch)  # Run evaluation per `eval_gap` batches (If so, save the best model)
    SAVE_AFTER_EPOCH = bool(args.save_after_epoch)  # Save the ckpt and results after each epoch
    EVAL_GAP = int(args.eval_gap)  # Run evaluation per `EVAL_GAP` batches
    LOGGING_GAP = int(args.logging_gap)  # Show loss per `LOGGING_GAP` batches
    EPOCH = int(args.epoch)  # The number of epochs for training
    BSZ_TRAIN = int(args.bsz_train)  # The batch size for training
    BSZ_GEN = int(args.bsz_gen)  # The batch size for generation /  evaluation
    INIT_LR = float(args.init_lr)  # The initial learning rate for training
    W_DECAY = float(args.w_decay)  # The weight decay rate for training
    N_ICL = int(args.n_icl)  # The number of examples for in-context learning
    N_GEN = int(args.n_gen)  # The number of sentences to be generated (for each evaluate(...) call)
    LEN_GEN = int(args.len_gen)  # The number of max tokens to be generated
    SAVE_DIR = str(args.save_dir)  # The directory of the current run
    CACHE_DIR = str(args.cache_dir)  # The directory where data & model are cached
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    LOG_DIR = str(args.log_dir)  # The directory to save logs
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    CKPT_DIR = str(args.ckpt_dir)  # The directory to save model checkpoints
    if not os.path.isdir(CKPT_DIR):
        os.makedirs(CKPT_DIR, exist_ok=True)
    OUTPUT_DIR = str(args.output_dir)  # The directory to outputs, e.g., results
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
    GPUS = CUDA.split(",") if "," in CUDA else [CUDA]
    GPUS = [int(gpu_id) for gpu_id in GPUS]
    if VERBOSE:
        logger.info(f"HAS_CUDA: {HAS_CUDA}; DEVICE: {DEVICE}; GPUS: {GPUS}")
        logger.info("torch.__version__:", torch.__version__)
        logger.info("torch.version.cuda:", torch.version.cuda)
        logger.info("torch.backends.cudnn.version():", torch.backends.cudnn.version())
        logger.info("torch.cuda.is_available():", torch.cuda.is_available())
        logger.info("torch.cuda.device_count():", torch.cuda.device_count())
        logger.info("torch.cuda.get_arch_list():", torch.cuda.get_arch_list())
        if HAS_CUDA:
            logger.info("torch.cuda.current_device():", torch.cuda.current_device())
            logger.info("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))

    run(verbose=VERBOSE)

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
