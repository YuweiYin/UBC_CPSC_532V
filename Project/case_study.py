import os
import gc
import re
import sys
import time
import json
import logging
import argparse
from functools import partial
from pathlib import Path
from typing import Union, List

import numpy as np
import torch.cuda

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.logging_utils import WandbLogger
from lm_eval.tasks import TaskManager, include_path, initialize_tasks
from lm_eval.utils import make_table

from util.TextParser import TextParser
from rag.retriever import (
    AtomicRetriever, GPTRetriever, WikiRetriever, ConceptNetRetriever, ArxivRetriever, GoogleSearchRetriever,
)
from rag.llm_agent import LLMAgent
from rag.prompt import (
    DirectQAPrompts, PreProcessingPrompts, PostProcessingPrompts, AugmentationPrompts, BackupPrompts,
)


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _int_or_none_list_arg_type(max_len: int, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items != max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )

    return items


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and "
             "log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument("--decontamination_ngrams_path", default=None)  # TODO: not used
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. "
             "Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations "
             "for comprehensive log output.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3),
        default="0,1234,1234",  # for backward compatibility
        help=(
            "Set seed for python's random, numpy and torch.\n"
            "Accepts a comma-separated list of 3 values for python's random, numpy, and torch seeds, respectively, "
            "or a single integer to set the same seed for all three.\n"
            "The values are either an integer or 'None' to not set the seed. Default is `0,1234,1234` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8` sets `random.seed(0)` and `torch.manual_seed(8)`. "
            "Here, the seed of numpy is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all three seeds to 42."
        ),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="The directory where data & model are cached"
    )
    parser.add_argument(
        "--use_rag",
        action="store_true",
        default=False,
        help="Use Retrieval-Augmented Generation (RAG) or not"
    )
    parser.add_argument(
        "--rag_source",
        type=str,
        default="ALL",
        help="Options: \"wiki\", \"conceptNet\", \"arxiv\", \"googleSearch\", \"llm\", \"atomic\". Default: \"ALL\""
    )
    parser.add_argument(
        "--use_rag_preprocess",
        action="store_true",
        default=False,
        help="Pre-process the original query or not"
    )
    parser.add_argument(
        "--rag_preprocess_type",
        type=str,
        default="contextual_clarification",
        help="Options: \"keyword_extraction\", \"contextual_clarification\", \"relevance_filtering\", "
             "\"query_expansion\", \"information_structuring\", \"intent_clarification\". "
             "Default: \"contextual_clarification\""
    )
    parser.add_argument(
        "--use_rag_postprocess",
        action="store_true",
        default=False,
        help="Post-process the retrievals or not"
    )
    parser.add_argument(
        "--rag_postprocess_type",
        type=str,
        default="summarizing_documents",
        help="Options: \"ranking_documents\", \"summarizing_documents\", \"extracting_key_info\", "
             "\"refining_documents\", \"evaluating_documents\", \"identifying_conflict\", "
             "\"filter_duplication\", \"structured_format\". "
             "Default: \"summarizing_documents\""
    )
    parser.add_argument(
        "--rag_limit",
        type=int,
        default=-1,
        help="The limit of the number of retrieved documents per knowledge source"
    )
    parser.add_argument(
        "--llm_retriever_type",
        type=str,
        default="google",
        help="The LLM retriever type. Default: \"google\" (free). Other options: \"openai\", \"anthropic\""
    )
    parser.add_argument(
        "--llm_agent_type",
        type=str,
        default="google",
        help="The LLM agent type. Default: \"google\" (free). Other options: \"openai\", \"anthropic\""
    )
    parser.add_argument(
        "--use_sft",
        action="store_true",
        default=False,
        help="Use supervised fine-tuning (via Instruction Tuning) or not"
    )
    parser.add_argument(
        "--use_icl",
        action="store_true",
        default=False,
        help="Use in-context learning (providing examples in the prompt) or not"
    )
    parser.add_argument(
        "--icl_n_example",
        type=int,
        default=3,
        help="The number of example to provide when using the ICL method."
    )
    parser.add_argument(
        "--use_cot",
        action="store_true",
        default=False,
        help="Use chain-of-thought prompting (providing reasoning path in the ICL examples) or not"
    )

    return parser.parse_args()


def run(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    args.cache_dir = str(args.cache_dir)  # The directory where data & model are cached
    if args.cache_dir == "":
        args.cache_dir = None
    else:
        if not os.path.isdir(args.cache_dir):
            os.makedirs(args.cache_dir, exist_ok=True)

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_name = str(args.model_args).strip()
    if "," in model_name:
        model_name = model_name.split(",")[0].strip()
    if "=" in model_name:
        model_name = model_name.split("=")[-1].strip()
    if "/" in model_name:
        model_name = model_name.split("/")[-1].strip()
    args.model_name = model_name

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        assert args.output_path, "Specify --output_path"

    initialize_tasks(args.verbosity)
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, cache_dir=args.cache_dir)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(task_manager.all_tasks))
        )
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, "
                    f"or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    args.output_path = os.path.join(args.output_path, args.model_name)
    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")

    eval_logger.info(f"Selected Tasks: {task_names}")
    eval_logger.info("Loading selected tasks...")

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

    # RAG settings
    args.use_rag = hasattr(args, "use_rag") and isinstance(args.use_rag, bool) and args.use_rag
    args.use_rag_preprocess = hasattr(args, "use_rag_preprocess") and \
        isinstance(args.use_rag_preprocess, bool) and args.use_rag_preprocess
    args.use_rag_postprocess = hasattr(args, "use_rag_postprocess") and \
        isinstance(args.use_rag_postprocess, bool) and args.use_rag_postprocess
    args.use_sft = hasattr(args, "use_sft") and isinstance(args.use_sft, bool) and args.use_sft
    args.use_icl = hasattr(args, "use_icl") and isinstance(args.use_icl, bool) and args.use_icl
    args.use_cot = hasattr(args, "use_cot") and isinstance(args.use_cot, bool) and args.use_cot
    N_EXAMPLE = int(args.icl_n_example)
    RAG_LIMIT = int(args.rag_limit)
    args.rag_source = str(args.rag_source) if hasattr(args, "rag_source") else "ALL"
    args.rag_preprocess_type = str(args.rag_preprocess_type) \
        if hasattr(args, "rag_preprocess_type") else "contextual_clarification"
    args.rag_postprocess_type = str(args.rag_postprocess_type) \
        if hasattr(args, "rag_postprocess_type") else "summarizing_documents"
    args.llm_retriever_type = str(args.llm_retriever_type) if hasattr(args, "llm_retriever_type") else "google"
    args.llm_agent_type = str(args.llm_agent_type) if hasattr(args, "llm_agent_type") else "google"

    if not args.use_rag:
        return

    # TextParser and Retrievers for Retrieval-Augmented Generation (RAG)
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
    # augmentationPrompts = AugmentationPrompts()
    # backupPrompts = BackupPrompts()
    # directQAAgent = LLMAgent(model=args.llm_agent_type)  # To answer the query directly. TODO: fit the lm_eval method
    preAgent = LLMAgent(model=args.llm_agent_type)  # To preprocess the query
    postAgent = LLMAgent(model=args.llm_agent_type)  # To postprocess the query and documents

    save_dir = str(args.output_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    timer_list = []

    def get_keywords(_query: str) -> List[str]:
        _keywords_1 = textParser.get_keywords_keybert(_query, n_bag=1)
        _keywords_2 = textParser.get_keywords_keybert(_query, n_bag=2)
        _keywords = sorted(list(set(_keywords_1 + _keywords_2)))
        return _keywords

    rag_timer_start = time.perf_counter()

    # ### RAG Step 1: Get cur_query of the current task
    # Different knowledge source API needs different query/keyword input
    cur_keywords = []
    cur_query = """
The fox walked from the city into the forest, what was it looking for?
    """.strip()  # This is an example from the CommonsenseQA benchmark

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
    # print(rag_docs)
    # for k, v in rag_docs.items():
    #     if len(v) > 0:
    #         print(f"\n{k}:")
    #         for _idx, _v in enumerate(v, start=1):
    #             print(f"{_idx}. {_v}")
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

    # ### RAG Step 4: Augmentation: Combine the docs to the original query (different prompting methods)
    if len(rag_all) > 0:
        rag_context = "Context:\n"
        for _idx, _doc in enumerate(rag_all, start=1):
            rag_context += f"{_idx}. {_doc}\n"
        # if len(atomic_rag) > 0:
        #     rag_context += "Atomic Knowledge:\n" + "\n".join(atomic_rag) + "\n"
        # if len(llm_rag) > 0:
        #     rag_context += "Large Language Model Knowledge:\n" + "\n".join(llm_rag) + "\n"
        # if len(wiki_rag) > 0:
        #     rag_context += "Wikipedia Knowledge:\n" + "\n".join(wiki_rag) + "\n"
        # if len(conceptNet_rag) > 0:
        #     rag_context += "ConceptNet Knowledge:\n" + "\n".join(conceptNet_rag) + "\n"
        # if len(arxiv_rag) > 0:
        #     rag_context += "arXiv Knowledge:\n" + "\n".join(arxiv_rag) + "\n"
        # if len(googleSearch_rag) > 0:
        #     rag_context += "Google Search Knowledge:\n" + "\n".join(googleSearch_rag) + "\n"
        rag_context += "\nAnswer the following query with the help of the above context:\n"
        rag_prompt = rag_context + cur_query

        # augmentation_short_prompt = augmentationPrompts.augmentation_short(cur_query, docs=rag_docs)
        # augmentation_medium_prompt = augmentationPrompts.augmentation_medium(cur_query, docs=rag_docs)
        # augmentation_long_prompt = augmentationPrompts.augmentation_long(cur_query, docs=rag_docs)
        # rag_prompt = augmentation_short_prompt

        print(rag_prompt)

    # ### RAG Step 4.5: [Optional] Supervised fine-tuning (SFT) / Instruction Tuning / Alignment via RLHF or DPO
    # Use train.py / train_dp.py / train_ddp.py to fine-tune models; Load the trained model by
    # setting `--model_args "pretrained=/path/to/huggingface_checkpoint,dtype=float" for eval.py

    # ### RAG Step 5: Run models and get evaluation results
    # run requests through model
    # resps = getattr(lm, req_type)(cloned_reqs)

    rag_timer_end = time.perf_counter()
    rag_time = rag_timer_end - rag_timer_start

    eval_logger.info(">>> Requests Dealing Time (Per): %.1f sec (%.1f min)" % (rag_time, rag_time / 60))
    eval_logger.info(">>> Pre-processing Time (Per): %.1f sec (%.1f min)" % (pre_time, pre_time / 60))
    eval_logger.info(">>> Post-processing Time (Per): %.1f sec (%.1f min)" % (post_time, post_time / 60))


if __name__ == "__main__":
    timer_start = time.perf_counter()

    run()

    timer_end = time.perf_counter()
    logging.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    # torch.cuda.empty_cache()
    # gc.collect()
    sys.exit(0)
