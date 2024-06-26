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
from typing import Union

import numpy as np
import torch.cuda

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.logging_utils import WandbLogger
from lm_eval.tasks import TaskManager, include_path, initialize_tasks
from lm_eval.utils import make_table


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
        "--rag_augmentation_type",
        type=str,
        default="basic",
        help="Options: \"basic\", \"short\", \"medium\", \"long\". Default: \"basic\""
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


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    if args.wandb_args:
        wandb_logger = WandbLogger(args)

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

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=args.cache_requests
    )

    results = evaluator.simple_evaluate(
        args=args,
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        predict_only=args.predict_only,
        **request_caching_args,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        cache_dir=args.cache_dir,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            # results, indent=2, default=_handle_non_serializable, ensure_ascii=False
            results, indent=None, default=_handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        if args.output_path:
            output_path_file.open("w", encoding="utf-8").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", args.model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=None,  # indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), "
            f"limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()


if __name__ == "__main__":
    timer_start = time.perf_counter()

    cli_evaluate()

    timer_end = time.perf_counter()
    logging.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    torch.cuda.empty_cache()
    gc.collect()
    sys.exit(0)
