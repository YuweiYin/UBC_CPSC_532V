import time
import copy
import json
import collections
import itertools
import logging
import os.path
import random
from typing import TYPE_CHECKING, Optional, Union, List
# import multiprocessing as mp

import numpy as np
import torch

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.models
from lm_eval.evaluator_utils import (
    consolidate_results, get_sample_size, get_task_list, prepare_print_tasks, print_writeout, run_task_tests,
)
from lm_eval.logging_utils import add_env_info, get_git_commit_hash
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    eval_logger, positional_deprecated, simple_parse_args_string,
)
from lm_eval.caching.cache import delete_cache
from lm_eval.models.huggingface import HFLM

from util.TextParser import TextParser
from rag.retriever import (
    AtomicRetriever, GPTRetriever, WikiRetriever, ConceptNetRetriever, ArxivRetriever, GoogleSearchRetriever,
)
from rag.llm_agent import LLMAgent
from rag.prompt import (
    DirectQAPrompts, PreProcessingPrompts, PostProcessingPrompts, AugmentationPrompts, BackupPrompts,
)

if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.tasks import Task


@positional_deprecated
def simple_evaluate(
        args,
        model,
        model_args: Optional[Union[str, dict, None]] = None,
        tasks=None,
        num_fewshot: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        use_cache: Optional[str] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        decontamination_ngrams_path=None,
        write_out: bool = False,
        log_samples: bool = True,
        gen_kwargs: str = None,
        task_manager: TaskManager = None,
        verbosity: str = "INFO",
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        cache_dir: Optional[str] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param args
       Arguments
    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects.
        Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing),
        If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param decontamination_ngrams_path:
        decontamination_ngrams_path
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param task_manager: TaskManager
        task_manager
    :param verbosity: str
        verbosity
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param cache_dir: Optional[str] = None
        Directory to save Hugging Face datasets, models, and tokenizers. None: ~/.cache/huggingface/

    :return
        Dictionary of results
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    assert (
            tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    lm = None
    if isinstance(model, str):
        if model_args is None:
            model_args = ""

        elif isinstance(model_args, dict):
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                    "cache_dir": cache_dir,
                },
            )

        else:
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                    "cache_dir": cache_dir,
                },
            )
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    if use_cache is not None:
        eval_logger.info(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    eval_logger.info(
        "get_task_dict has been updated to accept an optional argument, `task_manager`. Read more here: "
        "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage"
    )
    task_dict = get_task_dict(tasks, task_manager, cache_dir=cache_dir)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
            if task_obj is None:
                continue

        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )

        if predict_only:
            log_samples = True
            eval_logger.info(
                f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
            )
            # we have to change the class properties post-hoc. This is pretty hacky.
            task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. "
                    f"Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    # Show LM model
    if lm is None:
        raise ValueError(f"lm is None!")
    if isinstance(lm, HFLM):
        param_all = sum(p.numel() for p in lm.model.parameters())
        param_train = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
    elif hasattr(lm, "lm") and isinstance(lm.lm, HFLM):
        param_all = sum(p.numel() for p in lm.lm.model.parameters())
        param_train = sum(p.numel() for p in lm.lm.model.parameters() if p.requires_grad)
    else:
        raise ValueError(f"ValueError: type(lm) = {type(lm)}")
    eval_logger.info(f"Model parameters: ALL = {param_all}; Trainable = {param_train}")

    results = evaluate(
        args=args,
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
        verbosity=verbosity,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": (
                list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
            ),
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
            "cache_dir": cache_dir,
        }
        results["git_hash"] = get_git_commit_hash()
        add_env_info(results)  # additional environment info to results
        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
        args,
        lm: "LM",
        task_dict,
        limit: Optional[int] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        bootstrap_iters: Optional[int] = 100000,
        decontamination_ngrams_path=None,
        write_out: bool = False,
        log_samples: bool = True,
        verbosity: str = "INFO",
        verbose: bool = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param args
        Arguments
    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param cache_requests: bool.
    :param rewrite_requests_cache: bool.
    :param bootstrap_iters: Optional[int]
        Number of iterations for bootstrap statistics
    :param decontamination_ngrams_path
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param verbosity: str
    :param verbose: bool
    :return
        Dictionary of results
    """

    # RAG settings
    args.use_rag = hasattr(args, "use_rag") and isinstance(args.use_rag, bool) and args.use_rag
    args.use_rag_preprocess = hasattr(args, "use_rag_preprocess") and \
                              isinstance(args.use_rag_preprocess, bool) and args.use_rag_preprocess
    args.use_rag_postprocess = hasattr(args, "use_rag_postprocess") and \
                               isinstance(args.use_rag_postprocess, bool) and args.use_rag_postprocess
    args.use_sft = hasattr(args, "use_sft") and isinstance(args.use_sft, bool) and args.use_sft
    args.use_icl = hasattr(args, "use_icl") and isinstance(args.use_icl, bool) and args.use_icl
    args.icl_n_example = hasattr(args, "icl_n_example") and isinstance(args.icl_n_example, int) and args.icl_n_example
    args.use_cot = hasattr(args, "use_cot") and isinstance(args.use_cot, bool) and args.use_cot
    N_EXAMPLE = int(args.icl_n_example)
    RAG_LIMIT = int(args.rag_limit)
    args.rag_source = str(args.rag_source) if hasattr(args, "rag_source") else "ALL"
    args.rag_preprocess_type = str(args.rag_preprocess_type) \
        if hasattr(args, "rag_preprocess_type") else "contextual_clarification"
    args.rag_postprocess_type = str(args.rag_postprocess_type) \
        if hasattr(args, "rag_postprocess_type") else "summarizing_documents"
    args.rag_augmentation_type = str(args.rag_augmentation_type) \
        if hasattr(args, "rag_augmentation_type") else "basic"
    args.llm_retriever_type = str(args.llm_retriever_type) if hasattr(args, "llm_retriever_type") else "google"
    args.llm_agent_type = str(args.llm_agent_type) if hasattr(args, "llm_agent_type") else "google"

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    # decontaminate = decontamination_ngrams_path is not None

    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)

    # Few-shot for In-context learning and Chain-of-Thought prompting
    for k, v in task_dict.items():
        v.config.num_fewshot = max(0, args.icl_n_example) if args.use_icl else None

    # get lists of group hierarchy and each type of request
    task_hierarchy, eval_tasks = get_task_list(task_dict)
    if not log_samples:
        assert all(
            "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
            for task_output in eval_tasks
        ), "log_samples must be True for 'bypass' only tasks"
    for task_output in eval_tasks:
        task: Task = task_output.task
        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
        )
        eval_logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )

        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            req_type = instance.request_type
            requests[req_type].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )

            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[task.OUTPUT_TYPE] += numpad

    # ########## RAG ##########
    # Retrieval-Augmented Generation (RAG) workflow
    #     Step 1: Get cur_query of the current task
    #     Step 1.5: [Optional] Query preprocessing, e.g., rewriting
    #     Step 2: Keywords extraction (using KeyBERT or a GPT agent)
    #     Step 3: Search for relevant documents from multiple knowledge bases/sources (using online API)
    #     Step 3.5: [Optional] Document postprocessing, e.g., ranking, refinement, summarization, etc.
    #     Step 4: Augmentation: Combine the documents to the original query (different prompting methods)
    #     Step 4.5: [Optional] Supervised fine-tuning (SFT) / Instruction Tuning / Alignment via RLHF or DPO
    #     Step 5: Run models and get evaluation results

    # TextParser and Retrievers for Retrieval-Augmented Generation (RAG)
    if args.use_rag:
        textParser = TextParser()  # Keyword extractor
        wikiRetriever = WikiRetriever(full_text=False)
        googleSearchRetriever = GoogleSearchRetriever()
        LLMRetriever = LLMAgent(model=args.llm_retriever_type)  # LLM (Default: Google Gemini. Free)
        conceptNetRetriever = ConceptNetRetriever(verbose=False)
        arxivRetriever = ArxivRetriever()
        # atomicRetriever = AtomicRetriever()  # Too slow, and of questionable quality

    # Prompts (templates) and agents for LLMs
    # directQAPrompts = DirectQAPrompts()
    prePrompts = PreProcessingPrompts()
    postPrompts = PostProcessingPrompts()
    augmentationPrompts = AugmentationPrompts()
    # backupPrompts = BackupPrompts()
    # directQAAgent = LLMAgent(model=args.llm_agent_type)  # To answer the query directly. TODO: fit the lm_eval method
    preAgent = LLMAgent(model=args.llm_agent_type)  # To preprocess the query
    postAgent = LLMAgent(model=args.llm_agent_type)  # To postprocess the query and documents

    save_dir = str(args.output_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    req_timer_list = []
    rag_timer_list = []
    pre_timer_list = []
    post_timer_list = []

    def get_keywords(_query: str) -> List[str]:
        _keywords_1 = textParser.get_keywords_keybert(_query, n_bag=1)
        _keywords_2 = textParser.get_keywords_keybert(_query, n_bag=2)
        _keywords = sorted(list(set(_keywords_1 + _keywords_2)))
        return _keywords

    # Execute each type of request
    for req_type, reqs in requests.items():
        eval_logger.info(f"Running {req_type} requests")

        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        reqs_to_save = []
        for req in reqs:
            req_timer_start = time.perf_counter()

            # Evaluator workflow
            #     The evaluator feeds the model with req.arguments[0] (called "context") as input and
            #     compares the model output with req.arguments[0] (called "continuation"), which calculate two values:
            #     1. the logits.sum() of the generation probability;
            #     2. the exact match score (whether the model generation is exactly the same as the reference)
            #     See lm_eval/api/model.py `rem_res = getattr(self.lm, attr)(remaining_reqs)` and `loglikelihood(...)`
            #     See lm_eval/models/huggingface.py `_loglikelihood_tokens(...)` and `multi_logits = F.log_softmax(...)`
            #     Therefore, we need to prepend external knowledge (RAG retrievals) to the front of req.arguments[0]
            req.arguments_original = copy.deepcopy(req.arguments)  # Tuple(str): constructed input-output prompt
            # req_docs = req.doc  # dict: task-specific dictionary -> raw attributes to construct our RAG prompt

            # ### RAG Step 1: Get cur_query of the current task
            # Different knowledge source API needs different query/keyword input
            task_name = req.task_name
            cur_keywords = []
            match task_name:
                case "wsc273":
                    # The Winograd Schema Challenge  https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html
                    # req.arguments: Tuple(str, str)  # the same for all tasks
                    # req.doc: {"text": str, "pronoun": str, "pronoun_loc": int, "quote": str, "quote_loc": int,
                    #     "options": List[str], "label": int, "source": str}
                    # Task: Pick an option in `options` that the `pronoun` at `pronoun_loc` in the `text` refers to
                    cur_query = req.doc["text"]
                case "winogrande":
                    # WinoGrande: Adversarial WSC  https://winogrande.allenai.org/
                    # req.doc: {"sentence": str, "option1": str, "option2": str, "answer": str}
                    # Task: Pick either `option1` or `option2` to replace the "_" in the `sentence`
                    cur_query = req.doc["sentence"]
                    # answer_id = req.doc["answer"]
                    # if answer_id == "1":
                    #     answer_text = req.doc["option1"]
                    # elif answer_id == "2":
                    #     answer_text = req.doc["option2"]
                    # else:
                    #     raise ValueError(f"ValueError: answer = {answer_id}")
                    # cur_query = req.doc["sentence"].replace("_", answer_text)
                case "anli_r1":  # "anli"
                    # Adversarial NLI  https://github.com/facebookresearch/anli
                    # req.doc: {"uid": str, "premise": str, "hypothesis": str, "label": int, "reason": str}
                    # Task: NLI (premise -> hypothesis): label = 0 entailment; label = 1 neutral; label = 2 contradict
                    cur_query = req.doc["hypothesis"]
                case "anli_r2":  # "anli"
                    cur_query = req.doc["hypothesis"]
                case "anli_r3":  # "anli"
                    cur_query = req.doc["hypothesis"]
                case "arc_easy":  # "ai2_arc"
                    # AI2 Reasoning Challenge  https://allenai.org/data/arc
                    # req.doc: {"id": str, "question": str, "answerKey": str,
                    #     "choices": dict{"text": list, label: list}}
                    # Task: Answer the question by picking a choice (label) from the choices
                    cur_query = req.doc["question"]
                case "arc_challenge":  # "ai2_arc"
                    cur_query = req.doc["question"]
                case "piqa":
                    # Physical Interaction QA  https://leaderboard.allenai.org/physicaliqa/submissions/public
                    # req.doc: {"goal": str, "sol1": str, "sol2": str, "label": int}
                    # Task: Answer the question (`goal`) by picking a choice from `sol1` and `sol2`
                    cur_query = req.doc["goal"]
                case "swag":
                    # Situations With Adversarial Generations  https://rowanzellers.com/swag/
                    # req.doc: {"startphrase": str, "sent1": str, "sent2": str,
                    #     "ending0": str, "ending1": str, "ending2": str, "ending3": str, "label": int}
                    # Task: Complete the sentence (`startphrase`) by picking a choice from `ending0`--`ending3`
                    cur_query = req.doc["sent1"]
                case "hellaswag":
                    # Harder SWAG for Commonsense NLI https://rowanzellers.com/hellaswag/
                    # req.doc: {"query": str, "ctx": str, "ctx_a": str, "ctx_b": str, "activity_label": str,
                    #     "endings": List[str], "choices": List[str], "label": str, "gold": int, "split": str}
                    # Task: Complete the sentence (`query`) by picking a choice from the `endings` (or `choices`) list
                    cur_query = req.doc["ctx_a"]
                case "rte":  # GLUE - "glue"  https://gluebenchmark.com/
                    # Recognizing Textual Entailment
                    # req.doc: {"sentence1": str, "sentence2": str, "label": int, "idx": int}
                    # Task: Predict if `sentence1` entails `sentence1`. 0 -> entailment; 1 -> not entailment
                    cur_query = req.doc["sentence1"] + " " + req.doc["sentence2"]
                case "qnli":  # GLUE
                    # Stanford Question Answering Dataset
                    # req.doc: {"question": str, "sentence": str, "label": int, "idx": int}
                    # Task: Predict if `sentence` answer the `question`. label=0 -> yes; label=1 -> no
                    cur_query = req.doc["question"]
                    # cur_query = req.doc["question"] + " " + req.doc["sentence"]
                case "mnli":  # GLUE
                    # Multi-Genre Natural Language Inference Corpus
                    # req.doc: {"premise": str, "hypothesis": str, "label": int, "idx": int}
                    # Task: Predict if `premise` entails `hypothesis`. 0 -> contradict; 1 -> neutral; 2 -> entailment
                    cur_query = req.doc["premise"]
                    # cur_query = req.doc["premise"] + " " + req.doc["hypothesis"]
                case "mnli_mismatch":  # GLUE
                    cur_query = req.doc["premise"]
                    # cur_query = req.doc["premise"] + " " + req.doc["hypothesis"]
                case "mrpc":  # GLUE
                    # Microsoft Research Paraphrase Corpus
                    # req.doc: {"sentence1": str, "sentence2": str, "label": int, "idx": int}
                    # Task: Predict if `sentence1` and `sentence2` is equivalent. label=0 -> no; 1 -> yes
                    cur_query = req.doc["sentence1"] + " " + req.doc["sentence2"]
                case "qqp":  # GLUE
                    # Quora Question Pairs2 dataset
                    # req.doc: {"question1": str, "question2": str, "label": int, "idx": int}
                    # Task: Determine whether a pair of questions are semantically equivalent. label=0 -> no; 1 -> yes
                    cur_query = req.doc["question1"] + " " + req.doc["question2"]
                case "wnli":  # GLUE
                    # Winograd Schema Challenge
                    # req.doc: {"sentence1": str, "sentence2": str, "label": int, "idx": int}
                    # Task: Predict if `sentence2` with the pronoun replaced is entailed by `sentence1`. 0->no; 1->yes
                    cur_query = req.doc["sentence2"]
                case "sst2":  # GLUE
                    # Stanford Sentiment Treebank
                    # req.doc: {"sentence": str, "label": int, "idx": int}
                    # Task: Predict the sentiment of a given sentence. label=0 -> negative; label=1 -> positive
                    cur_query = req.doc["sentence"]
                case "cola":  # GLUE
                    # Corpus of Linguistic Acceptability
                    # req.doc: {"sentence": str, "label": int, "idx": int}
                    # Task: Determine whether `sentence` is a grammatically correct English sentence.
                    cur_query = req.doc["sentence"]
                case "cb":  # SuperGLUE - "super-glue-lm-eval-v1"  https://super.gluebenchmark.com/
                    # CommitmentBank
                    # req.doc: {"premise": str, "hypothesis": str, "label": int, "idx": int}
                    # Task: Predict if `premise` entails `hypothesis`. 0 -> entailment; 1 -> contradict; 2 -> neutral
                    cur_query = req.doc["premise"]
                    # cur_query = req.doc["premise"] + " " + req.doc["hypothesis"]
                case "wic":  # SuperGLUE
                    # Word-in-Context
                    # req.doc: {"word": str, "sentence1": str, "start1": int, "end1": int,
                    #     "sentence2": str, "start2": int, "end2": int, "label": int, "idx": int}
                    # Task: Determine whether the word is used with the same sense in both sentences. 0 -> no; 1 -> yes
                    cur_query = req.doc["word"]
                    # cur_query = req.doc["word"] + ". " + req.doc["sentence1"]+ " " + req.doc["sentence2"]
                case "sglue_rte":  # SuperGLUE
                    # Recognizing Textual Entailment (RTE)
                    # req.doc: {"premise": str, "hypothesis": str, "label": int, "idx": int}
                    # Task: Predict whether the given premise entails the hypothesis. label=0 -> entailment; 1 -> not
                    cur_query = req.doc["premise"]
                    # cur_query = req.doc["premise"] + " " + req.doc["hypothesis"]
                case "boolq":  # SuperGLUE
                    # BoolQ (Boolean Questions)
                    # req.doc: {"question": str, "passage": str, "label": int, "idx": int}
                    # Task: Solve the QA task by answering the yes/no question about the passage. 0 -> False; 1 -> True
                    # cur_query = req.doc["passage"]
                    cur_query = req.doc["question"]
                    # cur_query = req.doc["passage"] + " " + req.doc["question"]
                case "copa":  # SuperGLUE
                    # Choice Of Plausible Alternatives
                    # req.doc: {"premise": str, "choice1": str, "choice2": str,
                    #     "question": str, "label": int, "idx": int}
                    # Task: Choose the alternative which has the more plausible causal relationship with the premise.
                    cur_query = req.doc["premise"]
                    # cur_query = req.doc["premise"] + ". " + req.doc["choice1"] + ". " + req.doc["choice2"]
                case "multirc":  # SuperGLUE
                    # Multi-Sentence Reading Comprehension
                    # req.doc: {"paragraph": str, "question": str, "answer": str, "label": int, "idx": dict}
                    # Task: A true/false question-answering task. label=0 -> False; 1 -> True.
                    cur_query = req.doc["question"]
                    # cur_query = req.doc["paragraph"] + ". " + req.doc["question"]
                case "record":  # SuperGLUE
                    # Reading Comprehension with Commonsense Reasoning Dataset
                    # RECORD is a multiple-choice QA task. Each example consists of a news article and
                    #     a Cloze-style question about the article in which one entity is masked out.
                    #     The system must predict the masked out entity from a given list of possible entities
                    #     in the provided passage, where the same entity may be expressed
                    #     using multiple different surface forms, all of which are considered correct.
                    # req.doc: {"passage": str, "query": str, "entities": List[str],
                    #     "entity_spans": Dict[str: List], "answers": List[str], "idx": dict}
                    # Task: Choose all feasible `entities` to replace `@placeholder` in the `query`
                    cur_query = req.doc["query"].replace("@placeholder", " ")
                    # cur_query = req.doc["passage"] + ". " + req.doc["query"]
                    cur_keywords = list(set(req.doc["entities"]))
                case "wsc":  # SuperGLUE
                    # Winograd Schema Challenge
                    # Previously, a version of WSC recast as NLI as included in GLUE, known as WNLI.
                    # SuperGLUE recasts the WSC dataset into the co-reference form.
                    # req.doc: {"text": str, "span1_text": str, "span1_index": int,
                    #     "span2_text": str, "span2_index": int, "label": int, "idx": int}
                    # Task: Predict if the pronoun `span2_text` refers to `span1_text`. 0->no; 1->yes
                    cur_query = req.doc["text"]
                case _:
                    raise ValueError(f"ValueError: task_name = {task_name}")

            # Idae: Solving most of the above tasks relies more on commonsense or inter-sentence reasoning instead of
            #     external factual knowledge based on semantic matching (either sentence-level or word-level).
            #     Therefore, the effectiveness of RAG for such tasks is unknown.
            #     At least, the "missing info" to solve the task is more of reasoning path/clues.
            #     It is also unknown if the provided extra knowledge will even degrade the performance.

            if args.use_rag:  # Use RAG
                rag_timer_start = time.perf_counter()

                # ### RAG Step 1.5: [Optional] Query preprocessing, e.g., rewriting
                if args.use_rag_preprocess:
                    pre_timer_start = time.perf_counter()

                    match args.rag_preprocess_type:
                        case "keyword_extraction":  # need formatting
                            preprocess_prompt = prePrompts.keyword_extraction(cur_query)
                        case "contextual_clarification":  # concise rewriting
                            preprocess_prompt = prePrompts.contextual_clarification(cur_query)
                        case "relevance_filtering":  # a bit more informative
                            preprocess_prompt = prePrompts.relevance_filtering(cur_query)
                        case "query_expansion":  # more informative
                            preprocess_prompt = prePrompts.query_expansion(cur_query)
                        case "information_structuring":  # structured output
                            preprocess_prompt = prePrompts.information_structuring(cur_query)
                        case "intent_clarification":  # concise rewriting
                            preprocess_prompt = prePrompts.intent_clarification(cur_query)
                        case _:
                            raise ValueError(f"ValueError: rag_preprocess_type = {args.rag_preprocess_type}")

                    pre_responses = preAgent.apply_agent(prompt=preprocess_prompt)
                    cur_query = pre_responses[0] if len(pre_responses) > 0 else ""

                    pre_timer_end = time.perf_counter()
                    pre_time = pre_timer_end - pre_timer_start
                else:
                    pre_time = 0.0
                pre_timer_list.append(pre_time)

                # ### RAG Step 2: Keywords extraction (using KeyBERT or a GPT agent)
                if len(cur_keywords) == 0:
                    cur_keywords = get_keywords(cur_query)
                    cur_keywords.sort()
                    cur_keywords = cur_keywords[: RAG_LIMIT] if RAG_LIMIT > 0 else cur_keywords
                    # cur_keywords_gpt = preAgent.apply_agent(prompt=keyword_extraction_prompt)

                # ### RAG Step 3: Search for relevant documents from multiple knowledge bases/sources (using online API)
                wiki_rag, googleSearch_rag, llm_rag, conceptNet_rag, arxiv_rag, atomic_rag = [], [], [], [], [], []
                match args.rag_source:
                    case "wiki":
                        wiki_rag = wikiRetriever.retrieve(cur_query)  # wiki pages of the concept
                        for kw in cur_keywords:  # searching using keywords
                            wiki_rag += wikiRetriever.retrieve(kw)
                        wiki_rag = [_doc.replace("\n", " ").strip() for _doc in wiki_rag]
                        wiki_rag = wiki_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else wiki_rag
                    case "googleSearch":
                        googleSearch_rag = googleSearchRetriever.retrieve(cur_query)  # Google Search top-N results
                        googleSearch_rag = [_doc.replace("\n", " ").strip() for _doc in googleSearch_rag]
                        googleSearch_rag = googleSearch_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else googleSearch_rag
                    case "llm":
                        llm_rag = LLMRetriever.apply_agent(cur_query)  # LLM
                        llm_rag = [_doc.replace("\n", " ").strip() for _doc in llm_rag]
                        llm_rag = llm_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else llm_rag
                    case "conceptNet":
                        conceptNet_rag = conceptNetRetriever.retrieve(cur_query)  # all the edges of the concept
                        for kw in cur_keywords:  # searching using keywords
                            conceptNet_rag += conceptNetRetriever.retrieve(kw)
                        conceptNet_rag = [_doc.replace("\n", " ").strip() for _doc in conceptNet_rag]
                        conceptNet_rag = conceptNet_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else conceptNet_rag
                    case "arxiv":
                        arxiv_rag = arxivRetriever.retrieve(cur_query)  # the Abstract of most relevant N papers
                        arxiv_rag = [_doc.replace("\n", " ").strip() for _doc in arxiv_rag]
                        arxiv_rag = arxiv_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else arxiv_rag
                    # case "atomic":  # atomicRetriever is the slowest (and it costs the most in memory)
                    #     atomic_rag = atomicRetriever.retrieve(cur_query)  # text completion by the Atomic-Comet model
                    #     atomic_rag = [_doc.replace("\n", " ").strip() for _doc in atomic_rag]
                    #     atomic_rag = atomic_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else atomic_rag
                    case "ALL":  # Skip atomic, conceptNet, and arxiv because they are too slow
                        wiki_rag = wikiRetriever.retrieve(cur_query)  # wiki pages of the concept
                        googleSearch_rag = googleSearchRetriever.retrieve(cur_query)  # Google Search top-N results
                        llm_rag = LLMRetriever.apply_agent(cur_query)  # LLM
                        # conceptNet_rag = conceptNetRetriever.retrieve(cur_query)  # all the edges of the concept
                        # arxiv_rag = arxivRetriever.retrieve(cur_query)  # the Abstract of most relevant N papers
                        # atomic_rag = atomicRetriever.retrieve(cur_query)  # text completion by the Atomic-Comet model

                        for kw in cur_keywords:  # searching using keywords
                            wiki_rag += wikiRetriever.retrieve(kw)
                            # conceptNet_rag += conceptNetRetriever.retrieve(kw)

                        wiki_rag = [_doc.replace("\n", " ").strip() for _doc in wiki_rag]
                        wiki_rag = wiki_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else wiki_rag
                        googleSearch_rag = [_doc.replace("\n", " ").strip() for _doc in googleSearch_rag]
                        googleSearch_rag = googleSearch_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else googleSearch_rag
                        llm_rag = [_doc.replace("\n", " ").strip() for _doc in llm_rag]
                        llm_rag = llm_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else llm_rag
                        # conceptNet_rag = [_doc.replace("\n", " ").strip() for _doc in conceptNet_rag]
                        # conceptNet_rag = conceptNet_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else conceptNet_rag
                        # arxiv_rag = [_doc.replace("\n", " ").strip() for _doc in arxiv_rag]
                        # arxiv_rag = arxiv_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else arxiv_rag
                        # atomic_rag = [_doc.replace("\n", " ").strip() for _doc in atomic_rag]
                        # atomic_rag = atomic_rag[: RAG_LIMIT] if RAG_LIMIT > 0 else atomic_rag
                    case _:
                        raise ValueError(f"ValueError: args.rag_source = {args.rag_source}")

                rag_docs = {
                    "wiki_rag": wiki_rag,
                    "googleSearch": googleSearch_rag,
                    "llm": llm_rag,
                    "conceptNet": conceptNet_rag,
                    "arxiv": arxiv_rag,
                    "atomic": atomic_rag,
                }
                rag_all = wiki_rag + googleSearch_rag + llm_rag + conceptNet_rag + arxiv_rag + atomic_rag

                # ### RAG Step 3.5: [Optional] Document postprocessing, e.g., ranking, refinement, summarization, etc.
                if args.use_rag_postprocess:
                    post_timer_start = time.perf_counter()

                    match args.rag_postprocess_type:
                        case "ranking_documents":  # Top 5 relevant docs
                            postprocess_prompt = postPrompts.ranking_documents(cur_query, docs=rag_all)
                        case "summarizing_documents":  # Summary of each doc
                            postprocess_prompt = postPrompts.summarizing_documents(cur_query, docs=rag_all)
                        case "extracting_key_info":  # Summary of all docs
                            postprocess_prompt = postPrompts.extracting_key_info(cur_query, docs=rag_all)
                        case "refining_documents":  # Summary of most docs
                            postprocess_prompt = postPrompts.refining_documents(cur_query, docs=rag_all)
                        case "evaluating_documents":  # Evaluation of each doc
                            postprocess_prompt = postPrompts.evaluating_documents(cur_query, docs=rag_all)
                        case "identifying_conflict":  # Agreements and Contradictions
                            postprocess_prompt = postPrompts.identifying_conflict(cur_query, docs=rag_all)
                        case "filter_duplication":  # Filtered docs
                            postprocess_prompt = postPrompts.filter_duplication(cur_query, docs=rag_all)
                        case "structured_format":  # Relevant info of each doc
                            postprocess_prompt = postPrompts.structured_format(cur_query, docs=rag_all)
                        case _:
                            raise ValueError(f"ValueError: rag_postprocess_type = {args.rag_postprocess_type}")

                    post_responses = postAgent.apply_agent(prompt=postprocess_prompt)
                    rag_all = post_responses[0].split("\n") if len(post_responses) > 0 else []
                    rag_all = [_doc.strip() for _doc in rag_all if len(_doc) > 0]

                    post_timer_end = time.perf_counter()
                    post_time = post_timer_end - post_timer_start
                else:
                    post_time = 0.0
                post_timer_list.append(post_time)

                # ### RAG Step 4: Augmentation: Combine the docs to the original query (different prompting methods)
                if len(rag_all) > 0:
                    match args.rag_augmentation_type:
                        case "basic":
                            aug_prompt = "Context:\n"
                            for _idx, _doc in enumerate(rag_all, start=1):
                                aug_prompt += f"{_idx}. {_doc}\n"
                                # if len(atomic_rag) > 0:
                                #     aug_prompt += "Atomic Knowledge:\n" + "\n".join(atomic_rag) + "\n"
                                # if len(llm_rag) > 0:
                                #     aug_prompt += "Large Language Model Knowledge:\n" + "\n".join(llm_rag) + "\n"
                                # if len(wiki_rag) > 0:
                                #     aug_prompt += "Wikipedia Knowledge:\n" + "\n".join(wiki_rag) + "\n"
                                # if len(conceptNet_rag) > 0:
                                #     aug_prompt += "ConceptNet Knowledge:\n" + "\n".join(conceptNet_rag) + "\n"
                                # if len(arxiv_rag) > 0:
                                #     aug_prompt += "arXiv Knowledge:\n" + "\n".join(arxiv_rag) + "\n"
                                # if len(googleSearch_rag) > 0:
                                #     aug_prompt += "Google Search Knowledge:\n" + "\n".join(googleSearch_rag) + "\n"
                            aug_prompt += "\nAnswer the following query with the help of the above context:\n"
                            aug_prompt += req.arguments[0]
                        case "short":
                            aug_prompt = augmentationPrompts.augmentation_short(cur_query, docs=rag_all)
                        case "medium":
                            aug_prompt = augmentationPrompts.augmentation_medium(cur_query, docs=rag_all)
                        case "long":
                            aug_prompt = augmentationPrompts.augmentation_long(cur_query, docs=rag_all)
                        case _:
                            raise ValueError(f"ValueError: rag_augmentation_type = {args.rag_augmentation_type}")

                    req.arguments = (aug_prompt, req.arguments[1])  # Augmentation

                rag_timer_end = time.perf_counter()
                rag_time = rag_timer_end - rag_timer_start
            else:
                rag_docs = {}
                rag_time = 0.0
            rag_timer_list.append(rag_time)

            cloned_reqs.extend([req] * req.repeats)
            req_to_save = {
                "task_name": req.task_name,
                "metadata": req.metadata,
                "repeats": req.repeats,
                "request_type": req.request_type,
                "idx": req.idx,
                "doc": req.doc,
                "arguments": req.arguments,
                "rag_docs": rag_docs,
            }
            # reqs_to_save.append(req_to_save)
            reqs_to_save.extend([req_to_save] * req.repeats)

            req_timer_end = time.perf_counter()
            req_time = req_timer_end - req_timer_start
            req_timer_list.append(req_time)

        req_timer_all, req_timer_avg = sum(req_timer_list), float(np.mean(req_timer_list))
        eval_logger.info(">>> Requests Time (ALL): %.1f sec (%.1f min)" % (req_timer_all, req_timer_all / 60))
        eval_logger.info(">>> Requests Time (AVG): %.1f sec (%.1f min)" % (req_timer_avg, req_timer_avg / 60))

        if args.use_rag:
            eval_logger.info(f">>> Use RAG. Knowledge sources: {args.rag_source}")
            rag_timer_all, rag_timer_avg = sum(rag_timer_list), float(np.mean(rag_timer_list))
            eval_logger.info(">>> >>> RAG Time (ALL): %.1f sec (%.1f min)" % (rag_timer_all, rag_timer_all / 60))
            eval_logger.info(">>> >>> RAG Time (AVG): %.1f sec (%.1f min)" % (rag_timer_avg, rag_timer_avg / 60))

            if args.use_rag_preprocess:
                eval_logger.info(f">>> Use RAG Pre-processing: {args.rag_preprocess_type}")
                pre_timer_all, pre_timer_avg = sum(pre_timer_list), float(np.mean(pre_timer_list))
                eval_logger.info(">>> >>> RAG Pre-processing Time (ALL): %.1f sec (%.1f min)" % (
                    pre_timer_all, pre_timer_all / 60))
                eval_logger.info(">>> >>> RAG Pre-processing Time (AVG): %.1f sec (%.1f min)" % (
                    pre_timer_avg, pre_timer_avg / 60))
            else:
                eval_logger.info(">>> Did NOT use RAG Pre-processing")

            if args.use_rag_postprocess:
                eval_logger.info(f">>> Use RAG Post-processing: {args.rag_postprocess_type}")
                post_timer_all, post_timer_avg = sum(post_timer_list), float(np.mean(post_timer_list))
                eval_logger.info(">>> >>> RAG Post-processing Time (ALL): %.1f sec (%.1f min)" % (
                    post_timer_all, post_timer_all / 60))
                eval_logger.info(">>> >>> RAG Post-processing Time (AVG): %.1f sec (%.1f min)" % (
                    post_timer_avg, post_timer_avg / 60))
            else:
                eval_logger.info(">>> Did NOT use RAG Post-processing")
        else:
            eval_logger.info(">>> Did NOT use RAG")

        if (lm.world_size > 1) and (padding_requests[req_type] > 0):
            req = reqs[-1]
            for _ in range(padding_requests[req_type]):
                cloned_reqs.extend([req] * req.repeats)

        # ### RAG Step 4.5: [Optional] Supervised fine-tuning (SFT) / Instruction Tuning / Alignment via RLHF or DPO
        # Use train.py / train_dp.py / train_ddp.py to fine-tune models; Load the trained model by
        # setting `--model_args "pretrained=/path/to/huggingface_checkpoint,dtype=float"` for eval.py

        # ### RAG Step 5: Run models and get evaluation results
        # run requests through model
        resps = getattr(lm, req_type)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for resp, req, req_to_save in zip(resps, cloned_reqs, reqs_to_save):
            req.resps.append(resp)
            req_to_save["resps"] = resp

        # Save reqs
        _task_name = str(reqs[-1].task_name).strip()
        if "/" in _task_name:
            _task_name = _task_name.split("/")[-1].strip()
        if args.use_rag:
            # save_req_fp = os.path.join(
            #     save_dir, f"{_task_name}---{args.model_name}---requests---rag_{args.rag_source}.jsonl")
            save_req_fp = os.path.join(save_dir, f"{_task_name}---requests---rag_{args.rag_source}.jsonl")
        else:
            # save_req_fp = os.path.join(save_dir, f"{_task_name}---{args.model_name}---requests.jsonl")
            save_req_fp = os.path.join(save_dir, f"{_task_name}---requests.jsonl")
        with open(save_req_fp, "w", encoding="utf-8") as fp_out:
            for req_to_save in reqs_to_save:
                to_write = json.dumps(req_to_save)
                fp_out.write(to_write + "\n")

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    # Outputs Post-processing
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        # Collect values of metrics on all datapoints
        # Unpack results and sort back in order and return control to Task
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(
                rank=RANK, limit=limit, world_size=WORLD_SIZE
            )
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)

    if WORLD_SIZE > 1:
        # if multi-gpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.logged_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            # then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    if RANK == 0:
        # Aggregate results over all datapoints: aggregate results; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        results, samples, configs, versions, num_fewshot = consolidate_results(
            eval_tasks
        )

        # Calculate group metrics
        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if len(task_list) == 0:
                    # task_hierarchy entries are either `group_name: [subtask1, subtask2, ...]` or `task_name: []`.
                    # We only want to operate on groups here.
                    continue
                metric_list = list(
                    {
                        key
                        for task in task_list
                        for key in results[task].keys()
                        if "_stderr" not in key and key not in ["alias", "samples"]
                    }
                )
                for metric in metric_list:
                    stderr = "_stderr,".join(metric.split(","))

                    # Gather metrics, sizes, and std_errs from subtasks
                    metrics = [
                        results[task][metric]
                        for task in task_list
                        if metric in results[task]
                    ]
                    std_errs = [
                        results[task][stderr]
                        for task in task_list
                        if stderr in results[task]
                    ]
                    sizes = [
                        results[task]["samples"]
                        for task in task_list
                        if metric in results[task]
                    ]

                    # Compute group's pooled metric and stderr
                    results[group][
                        metric
                    ] = lm_eval.api.metrics.aggregate_subtask_metrics(metrics, sizes)
                    if "N/A" in std_errs:
                        results[group][stderr] = "N/A"
                    else:
                        results[group][
                            stderr
                        ] = lm_eval.api.metrics.pooled_sample_stderr(std_errs, sizes)

                    results[group]["samples"] = sum(sizes)

        results_agg = collections.defaultdict(dict)
        groups_agg = collections.defaultdict(dict)
        all_tasks_list = list(task_hierarchy.keys())
        left_tasks_list = []
        while True:
            add_tasks_list = list(k for k in results_agg.keys())
            left_tasks_list = sorted(list(set(all_tasks_list) - set(add_tasks_list)))
            if len(left_tasks_list) == 0:
                break

            _task_hierarchy = {
                k: v for k, v in task_hierarchy.items() if k in left_tasks_list
            }
            _results_agg, _groups_agg = prepare_print_tasks(_task_hierarchy, results)

            results_agg = {**results_agg, **_results_agg}
            groups_agg = {**groups_agg, **_groups_agg}

        for group_name, task_list in task_hierarchy.items():
            if task_list:
                num_fewshot[group_name] = num_fewshot[
                    task_list[0]
                ]  # TODO: validate this

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
            "group_subtasks": {k: v for k, v in reversed(task_hierarchy.items())},
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        # Save results_dict
        for _task_name in results_dict["results"].keys():
            _task_name = str(_task_name).strip()
            if "/" in _task_name:
                _task_name = _task_name.split("/")[-1].strip()
            if args.use_rag:
                # save_res_fp = os.path.join(
                #     save_dir, f"{_task_name}---{args.model_name}---results---rag_{args.rag_source}.jsonl")
                save_res_fp = os.path.join(save_dir, f"{_task_name}---results---rag_{args.rag_source}.jsonl")
            else:
                # save_res_fp = os.path.join(save_dir, f"{_task_name}---{args.model_name}---results.jsonl")
                save_res_fp = os.path.join(save_dir, f"{_task_name}---results.jsonl")
            with open(save_res_fp, "w", encoding="utf-8") as fp_out:
                try:
                    fp_out.write(json.dumps({"results": results_dict["results"][_task_name]}) + "\n")
                    fp_out.write(json.dumps({"group_subtasks": results_dict["group_subtasks"][_task_name]}) + "\n")
                    fp_out.write(json.dumps({"configs": results_dict["configs"][_task_name]}) + "\n")
                    fp_out.write(json.dumps({"versions": results_dict["versions"][_task_name]}) + "\n")
                    fp_out.write(json.dumps({"n-shot": results_dict["n-shot"][_task_name]}) + "\n")
                    if "samples" in results_dict:
                        for _res in results_dict["samples"][_task_name]:
                            fp_out.write(json.dumps(_res) + "\n")
                except Exception as e:
                    if verbose:
                        eval_logger.info(e)
                    if os.path.isfile(save_res_fp):
                        os.remove(save_res_fp)
                    break

        return results_dict

    else:
        return None


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": (
            True if cache_requests == "true" or cache_requests == "refresh" else False
        ),
        "rewrite_requests_cache": True if cache_requests == "refresh" else False,
        "delete_requests_cache": True if cache_requests == "delete" else False,
    }

    return request_caching_args
