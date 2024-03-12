#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
import os
import sys
import time
import json
import random
import shutil
import logging
import argparse
from collections import deque

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel.distributed import DistributedDataParallel as dDP

from transformers import set_seed
import wandb

from dataset.DatasetLoader import DatasetLoader
from dataset.DatasetTorch import DatasetMultiChoiceQA
from model.ModelLoader import ModelLoader
from model.TokenizerLoader import TokenizerLoader


def training(
        cfg,
        ds_name: str,
        model_name: str,
        model,
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
    :param cfg: configuration / arguments.
    :param ds_name: the dataset name.
    :param model_name: the model name.
    :param model: the model to train.
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
    :param verbose: Verbose mode: show logs.
    :return: the tuned model and used tokenizer.
    """

    model.train()

    if isinstance(model, dDP):  # DDP training
        model.module.train()
        model = model.to(cfg.rank)
    else:  # Single GPU training
        model = model.to(cfg.device)

    if not isinstance(save_dir, str) or len(save_dir) == 0:
        if isinstance(model, dDP):
            save_dir = f"{ds_name}---{model_name}---ddp"
        else:
            save_dir = f"{ds_name}---{model_name}"

    save_ckpt_dir = os.path.join("runs", save_dir, cfg.ckpt_dir)
    if not os.path.isdir(save_ckpt_dir):
        os.makedirs(save_ckpt_dir, exist_ok=True)

    save_log_dir = os.path.join("runs", save_dir, cfg.log_dir)
    if not os.path.isdir(save_log_dir):
        os.makedirs(save_log_dir, exist_ok=True)

    # optimizer = optim.Adam(model.parameters(), lr=float(1e-3), weight_decay=float(5e-4))
    optimizer = optim.Adam(model.parameters(), lr=cfg.init_lr, weight_decay=cfg.w_decay)
    if verbose:
        cfg.logger.info(optimizer)

    if cfg.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        if verbose:
            cfg.logger.info(lr_scheduler)
    else:
        lr_scheduler = None

    all_losses = []  # store the loss of each batch (divided by epochs)
    all_valid_scores = []  # store the accuracy score on the valid set after evaluation (divided by epochs)
    all_test_scores = []  # store the accuracy score on the test set after evaluation (divided by epochs)
    best_valid_score = 0.0
    best_test_score = 0.0

    batch_cnt = 0
    for epoch in range(cfg.epoch):
        if verbose:
            cfg.logger.info(f">>> Start Epoch: {epoch} (RANK={cfg.rank})")
        epoch_losses = []
        period_losses = []
        period_start_time = time.perf_counter()

        for batch_idx, batch_train in enumerate(dataloader_train):
            # Raw inputs
            answer_list = batch_train["answer"]
            prompt_list = batch_train["prompt"]
            # prompt_list = [icl_prompt + prompt + "Answer: " for prompt in prompt_list]  # add ICL examples
            # prompt_list = [prompt + "Answer: " for prompt in prompt_list]  # without ICL examples
            prompt_list = [prompt + "Answer:" for prompt in prompt_list]  # without ICL examples nor trailing space
            assert len(prompt_list) == len(answer_list), f"Assertion Error: len(prompt_list) != len(answer_list)"
            input_list = [prompt + answer for prompt, answer in zip(prompt_list, answer_list)]

            # Tokenized inputs
            inputs = tokenizer_train(input_list, return_tensors="pt", padding=True)  # padding_side="right"
            inputs.data["labels"] = inputs.data["input_ids"].clone()  # inputs.input_ids.clone()
            if isinstance(model, dDP):
                inputs = inputs.to(cfg.rank)
            else:
                inputs = inputs.to(cfg.device)

            # Forward pass
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)  # use_cache=True
            # type(outputs) is <class 'transformers.modeling_outputs.CausalLMOutputWithCrossAttentions'>
            # last_hidden_states = outputs.hidden_states[-1]  # outputs.last_hidden_state
            loss = outputs.loss  # Language modeling loss (for next-token prediction)

            # Backpropagation
            optimizer.zero_grad()
            loss.sum().backward()  # DDP automatically synchronizes here
            optimizer.step()

            # loss_value = loss.detach().cpu().numpy().item()
            # loss_value = loss.sum().detach().cpu().numpy().item()
            loss_value = loss.mean().detach().cpu().numpy().item()
            epoch_losses.append(loss_value)

            # Training log
            if batch_cnt % logging_gap == 0:
                cur_log = f"[LOG] >>> Epoch {epoch} >>> Total Batch {batch_cnt + 1} >>> loss: {loss_value}"
                if verbose:
                    cfg.logger.info(cur_log)
                if cfg.use_wandb:
                    period_end_time = time.perf_counter()
                    period_duration = period_end_time - period_start_time
                    log_dict["time/train_period_duration"] = period_duration
                    log_dict["training/train_loss_current"] = loss_value
                    avg_period_losses = sum(period_losses) / len(period_losses) if len(period_losses) > 0 else 0.0
                    log_dict["training/train_loss_avg_period"] = avg_period_losses
                    log_dict["training/train_lr"] = optimizer.param_groups[0]["lr"]
                    # log_dict["training/weight_decay"] = optimizer.param_groups[0]["weight_decay"]
                    wandb.log(log_dict)

            # Run evaluation
            if do_eval_batch and batch_cnt % eval_gap == 0:
                if isinstance(dataloader_valid, DataLoader):
                    # Evaluation on the valid set during training
                    if verbose:
                        cfg.logger.info(f"[Evaluation - valid] >>> Epoch {epoch} >>> Total Batch {batch_cnt + 1} "
                                        f">>> loss: {loss}")
                    valid_score = evaluate(
                        cfg=cfg,
                        ds_name=ds_name,
                        model_name=model_name,
                        model=model,
                        tokenizer_eval=tokenizer_eval,
                        dataloader=dataloader_valid,
                        icl_prompt=icl_prompt,
                        save_dir=save_dir,
                        save_fn=f"results---valid---batch{batch_cnt + 1}_epoch{epoch}-RANK_{cfg.rank}.jsonl",
                        verbose=verbose,
                    )
                    all_valid_scores.append(valid_score)
                    if valid_score > best_valid_score:
                        best_valid_score = valid_score

                        if cfg.rank == 0:
                            # Save the best model based on the best valid score
                            save_ckpt_hf = os.path.join(save_ckpt_dir, "model_best")  # HF model folder
                            if not os.path.isdir(save_ckpt_hf):
                                os.makedirs(save_ckpt_hf, exist_ok=True)

                            if verbose:
                                cfg.logger.info(f"Save the best model (valid score = {valid_score}) at {save_ckpt_hf}")
                            if isinstance(model, dDP):
                                model.module.save_pretrained(save_ckpt_hf)
                            else:
                                model.save_pretrained(save_ckpt_hf)  # Save config.json and model.safetensors
                            # save_ckpt_torch_fp = os.path.join(save_ckpt_hf, f"torch_ckpt.pt")  # torch.save
                            # torch.save(model.state_dict(), save_ckpt_torch_fp)  # duplicated

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
                        cfg.logger.info(f"[Evaluation - test] >>> Epoch {epoch} >>> Total Batch {batch_cnt + 1} "
                                        f">>> loss: {loss}")
                    test_score = evaluate(
                        cfg=cfg,
                        ds_name=ds_name,
                        model_name=model_name,
                        model=model,
                        tokenizer_eval=tokenizer_eval,
                        dataloader=dataloader_test,
                        icl_prompt=icl_prompt,
                        save_dir=save_dir,
                        save_fn=f"results---test---batch{batch_cnt + 1}_epoch{epoch}-RANK_{cfg.rank}.jsonl",
                        verbose=verbose,
                    )
                    all_test_scores.append(test_score)
                    if test_score > best_test_score:
                        best_test_score = test_score

            batch_cnt += 1

        if cfg.use_lr_scheduler and lr_scheduler is not None:
            lr_scheduler.step()

        if verbose:
            cfg.logger.info(f">>> END of Epoch: {epoch} (RANK={cfg.rank})")
        # After each epoch, gather the training losses from all devices and then store them
        epoch_losses_tensors = [torch.zeros(len(epoch_losses), dtype=torch.float32).cuda(cfg.rank)
                                for _ in range(cfg.world_size)]
        epoch_losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32).cuda(cfg.rank)
        dist.all_gather(epoch_losses_tensors, epoch_losses_tensor, async_op=False)  # gather and sync
        epoch_losses = [ep_losses.tolist() for ep_losses in epoch_losses_tensors]  # List[Tensor] -> List[list]
        all_losses.append(epoch_losses)  # store
        avg_ep_loss = np.mean(epoch_losses)

        # Save the model with tokenizers at the end of this epoch
        if cfg.rank == 0 and save_after_epoch and cfg.ckpt_limit > 0:
            save_ckpt_hf = os.path.join(save_ckpt_dir, f"model_epoch_{epoch}")  # HF model folder
            if not os.path.isdir(save_ckpt_hf):
                os.makedirs(save_ckpt_hf, exist_ok=True)
            if verbose:
                cfg.logger.info(f"Save the model for epoch {epoch} at {save_ckpt_hf}")
            if isinstance(model, dDP):
                model.module.save_pretrained(save_ckpt_hf)
            else:
                model.save_pretrained(save_ckpt_hf)  # Save config.json and model.safetensors
            # save_ckpt_torch_fp = os.path.join(save_ckpt_hf, f"torch_ckpt.pt")  # torch.save
            # torch.save(model.state_dict(), save_ckpt_torch_fp)  # duplicated

            # Save the tokenizers
            save_tokenizer_train_hf = os.path.join(save_ckpt_hf, "tokenizer_train")
            if not os.path.isdir(save_tokenizer_train_hf):
                os.makedirs(save_tokenizer_train_hf, exist_ok=True)
            tokenizer_train.save_pretrained(save_tokenizer_train_hf)

            save_tokenizer_eval_hf = os.path.join(save_ckpt_hf, "tokenizer_eval")
            if not os.path.isdir(save_tokenizer_eval_hf):
                os.makedirs(save_tokenizer_eval_hf, exist_ok=True)
            tokenizer_train.save_pretrained(save_tokenizer_eval_hf)

            # Record the saving directory, delete directory if overflow
            cfg.ckpt_queue.append(save_ckpt_hf)
            if len(cfg.ckpt_queue) > cfg.ckpt_limit:
                overflow_dir = cfg.ckpt_queue.popleft()
                if os.path.isdir(overflow_dir):
                    shutil.rmtree(overflow_dir, ignore_errors=True)

        # Run evaluation at the end of this epoch
        if do_eval_epoch:
            if isinstance(dataloader_valid, DataLoader):
                # Evaluation on the valid set at the end of this epoch
                if verbose:
                    cfg.logger.info(f"[Evaluation - valid] >>> Epoch {epoch} >>> Total Batch {batch_cnt} "
                                    f">>> average loss: {avg_ep_loss}")
                valid_score = evaluate(
                    cfg=cfg,
                    ds_name=ds_name,
                    model_name=model_name,
                    model=model,
                    tokenizer_eval=tokenizer_eval,
                    dataloader=dataloader_valid,
                    icl_prompt=icl_prompt,
                    save_dir=save_dir,
                    save_fn=f"results---valid---batch{batch_cnt}_epoch{epoch}_END-RANK_{cfg.rank}.jsonl",
                    verbose=verbose,
                )
                all_valid_scores.append(valid_score)
                if valid_score > best_valid_score:
                    best_valid_score = valid_score

                    if cfg.rank == 0:
                        # Save the best model based on the best valid score
                        save_ckpt_hf = os.path.join(save_ckpt_dir, "model_best")  # HF model folder
                        if not os.path.isdir(save_ckpt_hf):
                            os.makedirs(save_ckpt_hf, exist_ok=True)
                        if verbose:
                            cfg.logger.info(f"Save the best model (valid score = {valid_score}) at {save_ckpt_hf}")
                        if isinstance(model, dDP):
                            model.module.save_pretrained(save_ckpt_hf)
                        else:
                            model.save_pretrained(save_ckpt_hf)  # Save config.json and model.safetensors
                        # save_ckpt_torch_fp = os.path.join(save_ckpt_hf, f"torch_ckpt.pt")  # torch.save
                        # torch.save(model.state_dict(), save_ckpt_torch_fp)  # duplicated

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
                    cfg.logger.info(f"[Evaluation - test] >>> Epoch {epoch} >>> Total Batch {batch_cnt} "
                                    f">>> average loss: {avg_ep_loss}")
                test_score = evaluate(
                    cfg=cfg,
                    ds_name=ds_name,
                    model_name=model_name,
                    model=model,
                    tokenizer_eval=tokenizer_eval,
                    dataloader=dataloader_test,
                    icl_prompt=icl_prompt,
                    save_dir=save_dir,
                    save_fn=f"results---test---batch{batch_cnt}_epoch{epoch}_END-RANK_{cfg.rank}.jsonl",
                    verbose=verbose,
                )
                all_test_scores.append(test_score)
                if test_score > best_test_score:
                    best_test_score = test_score

    # Save all the training losses and scores at the end of the training session
    if cfg.rank == 0:
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

    # wandb.finish()
    return {
        "model": model,
        "tokenizer_train": tokenizer_train,
        "tokenizer_eval": tokenizer_eval,
        "all_losses": all_losses,
        "all_valid_scores": all_valid_scores,
        "all_test_scores": all_test_scores,
        "best_valid_score": best_valid_score,
        "best_test_score": best_test_score,
    }


def evaluate(
        cfg,
        ds_name: str,
        model_name: str,
        model,
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
    Model generation for evaluation.
    :param cfg: configuration / arguments.
    :param ds_name: the dataset name.
    :param model_name: the model name.
    :param model: the model for generation.
    :param tokenizer_eval: the tokenizer for generation (left padding).
    :param dataloader: the dataloader for evaluation (mini-batched input/output).
    :param icl_prompt: the prefix prompt for in-context learning.
    :param do_forward: use forward pass to generate one next token for prediction. Otherwise, beam search OR sampling.
    :param choice_prob: if `do_forward`, whether to only consider the probabilities of the choices, like "A" "B" "C".
    :param save_dir: specified directory for saving the results.
    :param save_fn: specified filename for saving the results.
    :param verbose: Verbose mode: show logs.
    :return: Accuracy score. Save the evaluation results/scores after generation.
    """

    if cfg.rank != 0 and not cfg.ddp_gen:
        if cfg.verbose:
            cfg.logger.error(f"`--ddp_gen` is not enabled, so evaluate() is ONLY for RANK=0. Cur rank = {cfg.rank}.")
        return 0.0
    assert cfg.rank == 0 or cfg.ddp_gen

    model.eval()

    if isinstance(model, dDP):  # DDP generation
        model.module.eval()
        model = model.to(cfg.rank)
    else:  # Single GPU generation
        model = model.to(cfg.device)

    if not isinstance(save_dir, str) or len(save_dir) == 0:
        if isinstance(model, dDP):
            save_dir = f"{ds_name}---{model_name}---ddp"
        else:
            save_dir = f"{ds_name}---{model_name}"

    all_prompts, all_preds, all_answers = [], [], []

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
        inputs = tokenizer_eval(input_list, return_tensors="pt", padding=True)  # padding_side="left"
        # inputs.data["labels"] = inputs.data["input_ids"].clone()
        if isinstance(model, dDP):
            inputs = inputs.to(cfg.rank)
        else:
            inputs = inputs.to(cfg.device)

        max_input_len = int(inputs.input_ids.shape[-1])
        input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]
        # seq_lengths = attention_mask.sum(-1)
        # input_decode = tokenizer_eval.batch_decode(
        #     input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if do_forward:
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, output_attentions=True)  # use_cache=True
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
            if isinstance(model, dDP):
                generator = model.module
            else:
                generator = model

            with torch.no_grad():
                outputs_gen = generator.generate(
                    input_ids,
                    pad_token_id=tokenizer_eval.pad_token_id,
                    bos_token_id=tokenizer_eval.bos_token_id,
                    eos_token_id=tokenizer_eval.eos_token_id,
                    # max_length=512,
                    max_length=max_input_len + cfg.len_gen,
                    num_beams=5,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    num_return_sequences=cfg.n_gen,
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

    # After generation, gather all the results from all devices and then store them
    # if cfg.ddp_gen:
    #     # Tokenize the strings as tensors before gathering
    #     all_prompts_tensor = tokenizer_eval(all_prompts, return_tensors="pt", padding=True).input_ids.cuda(cfg.rank)
    #     all_prompts_tensors = [torch.zeros_like(all_prompts_tensor, dtype=torch.int64).cuda(cfg.rank)
    #                            for _ in range(cfg.world_size)]
    #     all_preds_tensor = tokenizer_eval(all_preds, return_tensors="pt", padding=True).input_ids.cuda(cfg.rank)
    #     all_preds_tensors = [torch.zeros_like(all_preds_tensor, dtype=torch.int64).cuda(cfg.rank)
    #                          for _ in range(cfg.world_size)]
    #     all_answers_tensor = tokenizer_eval(all_answers, return_tensors="pt", padding=True).input_ids.cuda(cfg.rank)
    #     all_answers_tensors = [torch.zeros_like(all_answers_tensor, dtype=torch.int64).cuda(cfg.rank)
    #                            for _ in range(cfg.world_size)]
    #     # Gather and synchronize
    #     dist.all_gather(all_prompts_tensors, all_prompts_tensor, async_op=False)
    #     dist.all_gather(all_preds_tensors, all_preds_tensor, async_op=False)
    #     dist.all_gather(all_answers_tensors, all_answers_tensor, async_op=False)
    #     # Decode the ids to the original strings
    #     all_prompts_list = [
    #         tokenizer_eval.batch_decode(prompts_tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #         for prompts_tensors in all_prompts_tensors
    #     ]  # list of all_prompts from all devices (ranks)
    #     all_preds_list = [
    #         tokenizer_eval.batch_decode(preds_tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #         for preds_tensors in all_preds_tensors
    #     ]
    #     all_answers_list = [
    #         tokenizer_eval.batch_decode(answers_tensors, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    #         for answers_tensors in all_answers_tensors
    #     ]
    #     # Flatten the List[List[str]] to List[str]
    #     all_prompts = [_p for _all in all_prompts_list for _p in _all]
    #     all_preds = [_p for _all in all_preds_list for _p in _all]
    #     all_answers = [_a for _all in all_answers_list for _a in _all]

    # Compute the accuracy
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

    res_len = len(results)
    accuracy = len(correct_idx) / res_len if res_len > 0 else 0.0

    # After computing the accuracy of the current device, gather/reduce the accuracies from all devices
    if cfg.ddp_gen:
        acc_tensor = torch.tensor(accuracy, dtype=torch.float32).cuda(cfg.rank)
        acc_tensors = [torch.zeros_like(acc_tensor, dtype=torch.float32).cuda(cfg.rank)
                       for _ in range(cfg.world_size)]
        dist.all_gather(acc_tensors, acc_tensor, async_op=False)  # gather and sync
        # accuracy = (sum(acc_tensors) / cfg.world_size).cpu().numpy().item() if cfg.world_size > 0 else 0.0
        # dist.all_reduce(acc_tensor, async_op=False)  # reduce and sync
        # accuracy = (acc_tensor / cfg.world_size).cpu().numpy().item() if cfg.world_size > 0 else 0.0

        len_tensor = torch.tensor(res_len, dtype=torch.float32).cuda(cfg.rank)
        len_tensors = [torch.zeros_like(len_tensor, dtype=torch.float32).cuda(cfg.rank)
                       for _ in range(cfg.world_size)]
        dist.all_gather(len_tensors, len_tensor, async_op=False)  # gather and sync
        # dist.all_reduce(len_tensor, async_op=False)  # reduce and sync

        assert len(acc_tensors) == len(len_tensors) == cfg.world_size
        weighted_sum = sum([acc * r_len for acc, r_len in zip(acc_tensors, len_tensors)])
        total_len = sum(len_tensors)
        accuracy = (weighted_sum / total_len).cpu().numpy().item() if total_len > 0 else 0.0

    if verbose:
        # cfg.logger.info(f">>> Accuracy (RANK={cfg.rank}): {accuracy:.5f}")
        cfg.logger.info(f">>> Accuracy (RANK={cfg.rank}): {accuracy}")

    # Save the evaluation results
    if cfg.rank == 0 or cfg.ddp_gen:
        save_results_dir = os.path.join("runs", save_dir, cfg.output_dir)
        if not os.path.isdir(save_results_dir):
            os.makedirs(save_results_dir, exist_ok=True)
        if isinstance(save_fn, str) and len(save_fn) > 0:
            save_results_path = os.path.join(save_results_dir, save_fn)
        else:
            save_results_path = os.path.join(
                save_results_dir, f"eval_results-accuracy_{accuracy:.5f}-RANK_{cfg.rank}.jsonl")
        with open(save_results_path, "w", encoding="utf-8") as fp_out:
            for result in results:
                fp_out.write(json.dumps(result) + "\n")

    return accuracy


def run(
        rank,
        world_size,
        cfg,
) -> None:
    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = cfg.master_port

    # initialize the process group
    dist.init_process_group(backend=cfg.backend, rank=rank, world_size=world_size)

    # rank = dist.get_rank()
    w_size = dist.get_world_size()
    assert w_size == world_size == cfg.world_size == cfg.device_count == len(cfg.gpus) > 0
    cfg.rank = rank

    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    cfg.logger = logging.getLogger(name=f"RANK_{rank}")
    cfg.logger.info(f"Running DDP on rank {rank}: world_size={world_size}; "
                    f"backend={cfg.backend}; master_addr={cfg.master_addr}; master_port={cfg.master_port}")

    cfg.verbose = cfg.verbose_all or (cfg.verbose and rank == 0)  # if not verbose_all, only rank=0 can output logs

    if cfg.verbose:
        cfg.logger.info(cfg)

    # Loaders
    datasetLoader = DatasetLoader(cfg.logger)
    modelLoader = ModelLoader(cfg.logger)
    tokenizerLoader = TokenizerLoader(cfg.logger)

    # Set the random seed of all modules (again, before loading data/model/tokenizer)
    set_seed(cfg.seed)

    # Dataset
    dataset_dict = datasetLoader.load_dataset(
        ds_name=cfg.ds_name, n_icl=cfg.n_icl, cache_dir=cfg.cache_dir, random_seed=cfg.seed, verbose=cfg.verbose)
    dataset_hf = dataset_dict["dataset_hf"]
    icl_prompt = dataset_dict["icl_prompt"]

    # Model
    model_dict = modelLoader.load_model(model_name=cfg.model_name, cache_dir=cfg.cache_dir, verbose=cfg.verbose)
    model = model_dict["model"]
    # model = dDP(module=model, device_ids=[rank], output_device=0, dim=0)
    model = dDP(module=model)
    model = model.to(rank)
    if cfg.verbose:
        cfg.logger.info(f"type(model) = {type(model)}")

    # Tokenizer
    tokenizer_train_dict = tokenizerLoader.load_tokenizer(
        model_name=cfg.model_name, is_train=True, cache_dir=cfg.cache_dir, verbose=cfg.verbose)
    tokenizer_train = tokenizer_train_dict["tokenizer"]
    tokenizer_eval_dict = tokenizerLoader.load_tokenizer(
        model_name=cfg.model_name, is_train=False, cache_dir=cfg.cache_dir, verbose=cfg.verbose)
    tokenizer_eval = tokenizer_eval_dict["tokenizer"]

    # Set the random seed of all modules (again, after loading data/model/tokenizer)
    set_seed(cfg.seed)

    # Dataset partition for each DDP process
    partition_ratio = 1.0 / w_size
    partition_ratios = [(i * partition_ratio, (i + 1) * partition_ratio) for i in range(w_size)]
    assert 0 <= rank < len(partition_ratios), f"Assertion Error: not 0 <= rank < len(partition_ratios): rank = {rank}"
    cur_ratio_range = partition_ratios[rank]
    if cfg.verbose:
        cfg.logger.info(f"partition_ratio = {partition_ratio}; cur_ratio_range = {cur_ratio_range}")

    # Convert Hugging Face datasets to PyTorch Dataset
    ds_torch_train = DatasetMultiChoiceQA(
        dataset=dataset_hf, tokenizer=tokenizer_train, splits="train", ratio_range=cur_ratio_range)
    if cfg.ddp_gen:  # valid and test set partition - generation/evaluation on multiple GPUs
        ds_torch_valid = DatasetMultiChoiceQA(
            dataset=dataset_hf, tokenizer=tokenizer_eval, splits="validation", ratio_range=cur_ratio_range)
        ds_torch_test = DatasetMultiChoiceQA(
            dataset=dataset_hf, tokenizer=tokenizer_eval, splits="test", ratio_range=cur_ratio_range)
    else:  # use all the valid and test set - generation/evaluation on the RANK=0 GPU
        ds_torch_valid = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_eval, splits="validation")
        ds_torch_test = DatasetMultiChoiceQA(dataset=dataset_hf, tokenizer=tokenizer_eval, splits="test")

    # PyTorch DataLoader (mini-batch)
    bsz_train_per_device = int(cfg.bsz_train / float(w_size))
    bsz_gen_per_device = int(cfg.bsz_gen / float(w_size))
    if cfg.verbose:
        cfg.logger.info(f"bsz_train_per_device = {bsz_train_per_device}; bsz_gen_per_device = {bsz_gen_per_device}")
    dataloader_train = DataLoader(ds_torch_train, batch_size=bsz_train_per_device, shuffle=False)
    dataloader_valid = DataLoader(ds_torch_valid, batch_size=bsz_gen_per_device, shuffle=False)
    dataloader_test = DataLoader(ds_torch_test, batch_size=bsz_gen_per_device, shuffle=False)

    if cfg.verbose:
        cfg.logger.info(f"ds_torch_train: ratio_range = {ds_torch_train.ratio_range}; "
                        f"index_range = {ds_torch_train.index_range}")
        cfg.logger.info(f"ds_torch_valid: ratio_range = {ds_torch_valid.ratio_range}; "
                        f"index_range = {ds_torch_valid.index_range}")
        cfg.logger.info(f"ds_torch_test: ratio_range = {ds_torch_test.ratio_range}; "
                        f"index_range = {ds_torch_test.index_range}")

    if cfg.eval_before:
        # Evaluation on the valid set before training
        if cfg.verbose:
            cfg.logger.info("Evaluation on the valid set before training...")
        evaluate(
            cfg=cfg,
            ds_name=cfg.ds_name,
            model_name=cfg.model_name,
            model=model,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_valid,
            icl_prompt=icl_prompt,
            save_dir=cfg.save_dir,
            save_fn=f"results_beforeFT_valid-RANK_{cfg.rank}.jsonl",
            verbose=cfg.verbose,
        )

        # Evaluation on the test set before training
        if cfg.verbose:
            cfg.logger.info("Evaluation on the test set before training...")
        evaluate(
            cfg=cfg,
            ds_name=cfg.ds_name,
            model_name=cfg.model_name,
            model=model,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_test,
            icl_prompt=icl_prompt,
            save_dir=cfg.save_dir,
            save_fn=f"results_beforeFT_test-RANK_{cfg.rank}.jsonl",
            verbose=cfg.verbose,
        )

    # Train (fine-tune) the model (Causal LM, next token prediction)
    if cfg.verbose:
        cfg.logger.info("Train (fine-tune) the model (Causal LM, next token prediction)...")
    ft_dict = training(
        cfg=cfg,
        ds_name=cfg.ds_name,
        model_name=cfg.model_name,
        model=model,
        tokenizer_train=tokenizer_train,
        tokenizer_eval=tokenizer_eval,
        dataloader_train=dataloader_train,
        dataloader_valid=dataloader_valid,
        dataloader_test=dataloader_test,
        icl_prompt=icl_prompt,
        do_eval_epoch=cfg.do_eval_epoch,
        do_eval_batch=cfg.do_eval_batch,
        save_after_epoch=cfg.save_after_epoch,
        eval_gap=cfg.eval_gap,
        logging_gap=cfg.logging_gap,
        save_dir=cfg.save_dir,
        verbose=cfg.verbose,
    )
    model_ft = ft_dict["model"]  # the trained model
    # tokenizer_train = ft_dict["tokenizer_train"]
    # tokenizer_eval = ft_dict["tokenizer_eval"]
    # all_losses = ft_dict["all_losses"]
    # all_valid_scores = ft_dict["all_valid_scores"]
    # all_test_scores = ft_dict["all_test_scores"]
    # best_valid_score = ft_dict["best_valid_score"]
    # best_test_score = ft_dict["best_test_score"]

    if cfg.eval_after:
        # Evaluation on the valid set before training
        if cfg.verbose:
            cfg.logger.info("Evaluation on the valid set after training...")
        evaluate(
            cfg=cfg,
            ds_name=cfg.ds_name,
            model_name=cfg.model_name,
            model=model_ft,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_valid,
            icl_prompt=icl_prompt,
            save_dir=cfg.save_dir,
            save_fn=f"results_afterFT_valid-RANK_{cfg.rank}.jsonl",
            verbose=cfg.verbose,
        )

        # Evaluation on the test set before training
        if cfg.verbose:
            cfg.logger.info("Evaluation on the test set after training...")
        evaluate(
            cfg=cfg,
            ds_name=cfg.ds_name,
            model_name=cfg.model_name,
            model=model_ft,
            tokenizer_eval=tokenizer_eval,
            dataloader=dataloader_test,
            icl_prompt=icl_prompt,
            save_dir=cfg.save_dir,
            save_fn=f"results_afterFT_test-RANK_{cfg.rank}.jsonl",
            verbose=cfg.verbose,
        )

    # if cfg.verbose:
    #     cfg.logger.info("Done!")
    cfg.logger.info(f"Done! [RANK={rank}]")

    # Do a dump forward pass for synchronizing all processes
    dump_inputs = tokenizer_train(icl_prompt.strip(), return_tensors="pt", padding=True)
    dump_inputs.data["labels"] = dump_inputs.data["input_ids"].clone()
    if isinstance(model, dDP):
        dump_inputs = dump_inputs.to(cfg.rank)
    else:
        dump_inputs = dump_inputs.to(cfg.device)
    outputs = model(**dump_inputs, output_hidden_states=False, output_attentions=False)
    loss = outputs.loss  # Language modeling loss (for next-token prediction)
    loss.sum().backward()  # Backpropagation (DDP automatically synchronizes here)

    time.sleep(3.0)
    # dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Verbose mode: show logs")
    parser.add_argument("--verbose_all", action="store_true", default=False, help="Verbose: show logs of all RANKs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of all modules")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA device(s), e.g., 0 OR 0,1")
    parser.add_argument("--ddp", action="store_true", default=False, help="Multi-GPU training")
    parser.add_argument("--ddp_gen", action="store_true", default=False, help="Multi-GPU generation")
    parser.add_argument("--backend", type=str, default="gloo", help="The DDP backend, e.g., gloo and nccl")
    parser.add_argument("--master_addr", type=str, default="localhost", help="The DDP communication address")
    parser.add_argument("--master_port", type=str, default="12345", help="The DDP communication port")
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
    parser.add_argument("--ckpt_limit", type=int, default=5, help="Limit the total amount of saved checkpoints")
    parser.add_argument("--eval_gap", type=int, default=1000, help="Run evaluation per `eval_gap` batches")
    parser.add_argument("--logging_gap", type=int, default=100, help="Show loss per `logging_gap` batches")
    parser.add_argument("--n_icl", type=int, default=5, help="The number of examples for in-context learning")
    parser.add_argument("--n_gen", type=int, default=1, help="The number of sentences to be generated")
    parser.add_argument("--len_gen", type=int, default=10, help="The number of max tokens to be generated")
    parser.add_argument("--epoch", type=int, default=5, help="The number of epochs for training")
    parser.add_argument("--bsz_train", type=int, default=32, help="The batch size for training")
    parser.add_argument("--bsz_gen", type=int, default=32, help="The batch size for generation / evaluation")
    parser.add_argument("--init_lr", type=float, default=float(1e-3), help="The initial learning rate for training")
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False, help="Use lr scheduler")
    # parser.add_argument("--w_decay", type=float, default=float(5e-4), help="The weight decay rate for training")
    parser.add_argument("--w_decay", type=float, default=0.0, help="The weight decay rate for training")
    parser.add_argument("--save_dir", type=str, default="", help="The directory of the current run")
    parser.add_argument("--cache_dir", type=str, default="~/.cache/huggingface/",
                        help="The directory where data & model are cached")
    parser.add_argument("--log_dir", type=str, default="log", help="The directory to save logs")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="The directory to save model checkpoints")
    parser.add_argument("--output_dir", type=str, default="output", help="The directory to outputs, e.g., results")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb to save & show logs")
    args = parser.parse_args()
    logger.info(args)

    timer_start = time.perf_counter()

    # Set the random seed of all modules
    args.seed = int(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    # Hyperparameters
    args.cuda = str(args.cuda).strip()  # CUDA device(s), e.g., "0" OR "0,1"
    args.ddp = bool(args.ddp)  # Multi-GPU training: DistributedDataParallel
    args.ddp_gen = bool(args.ddp_gen)  # Multi-GPU generation: DistributedDataParallel
    args.backend = str(args.backend)  # The DDP backend, e.g., nccl and gloo
    args.master_addr = str(args.master_addr)  # The DDP communication address
    args.master_port = str(args.master_port)  # The DDP communication port
    args.verbose = bool(args.verbose)  # Verbose mode: show logs
    args.verbose_all = bool(args.verbose)  # Verbose mode: show logs of all RANKs
    args.eval_before = bool(args.eval_before)  # Run evaluation before training
    args.eval_after = bool(args.eval_after)  # Run evaluation after training
    args.do_eval_epoch = bool(args.do_eval_epoch)  # Run evaluation after each epoch
    args.do_eval_batch = bool(args.do_eval_batch)  # Run evaluation per `eval_gap` batches (If so, save the best model)
    args.save_after_epoch = bool(args.save_after_epoch)  # Save the ckpt and results after each epoch
    args.ckpt_limit = max(0, int(args.ckpt_limit))  # Limit the total amount of saved checkpoints
    args.ckpt_queue = deque()  # The queue of `save_dir` of saved checkpoints (except the best model)
    args.eval_gap = int(args.eval_gap)  # Run evaluation per `EVAL_GAP` batches
    args.logging_gap = int(args.logging_gap)  # Show loss per `LOGGING_GAP` batches
    args.epoch = int(args.epoch)  # The number of epochs for training
    args.bsz_train = int(args.bsz_train)  # The batch size for training
    args.bsz_gen = int(args.bsz_gen)  # The batch size for generation /  evaluation
    args.init_lr = float(args.init_lr)  # The initial learning rate for training
    args.use_lr_scheduler = bool(args.use_lr_scheduler)  # Use learning rate scheduler (default: StepLR)
    args.w_decay = float(args.w_decay)  # The weight decay rate for training
    args.n_icl = int(args.n_icl)  # The number of examples for in-context learning
    args.n_gen = int(args.n_gen)  # The number of sentences to be generated (for each evaluate() call)
    args.len_gen = int(args.len_gen)  # The number of max tokens to be generated
    args.save_dir = str(args.save_dir)  # The directory of the current run
    args.cache_dir = str(args.cache_dir)  # The directory where data & model are cached
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)
    args.log_dir = str(args.log_dir)  # The directory to save logs
    args.ckpt_dir = str(args.ckpt_dir)  # The directory to save model checkpoints
    args.output_dir = str(args.output_dir)  # The directory to outputs, e.g., results
    args.use_wandb = bool(args.use_wandb)  # Use wandb to save & show logs

    # CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.has_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.has_cuda else "cpu")
    args.gpus = args.cuda.split(",") if "," in args.cuda else [args.cuda]
    args.gpus = [int(gpu_id) for gpu_id in args.gpus]
    args.device_count = int(torch.cuda.device_count())
    assert len(args.gpus) == args.device_count, \
        f"Assertion Error: len(gpus) = {len(args.gpus)}; device_count = {args.device_count}"
    args.world_size = args.device_count
    args.ddp_able = args.has_cuda and len(args.gpus) > 1
    if args.verbose:
        logger.info(f"HAS_CUDA: {args.has_cuda}; DEVICE: {args.device}; GPUS: {args.gpus}; "
                    f"DDP able: {args.ddp_able}; use DistributedDataParallel: {args.ddp}")
        logger.info(f"torch.__version__: {torch.__version__}")
        logger.info(f"torch.version.cuda: {torch.version.cuda}")
        logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        logger.info(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        logger.info(f"torch.cuda.get_arch_list(): {torch.cuda.get_arch_list()}")
        if args.has_cuda:
            logger.info(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
            logger.info(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    if args.ddp and args.ddp_able:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port

        mp.spawn(run,
                 args=(args.world_size, args),
                 nprocs=args.world_size,
                 join=True)
    else:
        logger.error(f"!!! Error: Please use multiple GPUs OR try `python3 main.py`")

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
