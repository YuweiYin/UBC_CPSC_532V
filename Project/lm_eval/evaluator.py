import collections
import itertools
import logging
import random
from typing import TYPE_CHECKING, Optional, Union, List

import numpy as np
import torch

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.models
from lm_eval.evaluator_utils import (
    consolidate_results,
    get_sample_size,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lm_eval.logging_utils import add_env_info, get_git_commit_hash
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    eval_logger,
    positional_deprecated,
    simple_parse_args_string,
)

import copy
from util.TextParser import TextParser
from rag.retriever import (
    AtomicRetriever, GPTRetriever, WikiRetriever, ConceptNetRetriever, ArxivRetriever, GoogleSearchRetriever,
)
from rag.prompt import (
    PROMPT, fastRAG_PROMPT, general_PROMPT,
    chatGPT_PROMPT_1, chatGPT_PROMPT_2, chatGPT_PROMPT_3, chatGPT_PROMPT_4, chatGPT_PROMPT_5,
)

if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.tasks import Task

from lm_eval.caching.cache import delete_cache


@positional_deprecated
def simple_evaluate(
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

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
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
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
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
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

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
        "get_task_dict has been updated to accept an optional argument, `task_manager`"
        "Read more here:https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage"
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
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
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
        lm: "LM",
        task_dict,
        limit: Optional[int] = None,
        cache_requests=False,
        rewrite_requests_cache=False,
        bootstrap_iters: Optional[int] = 100000,
        decontamination_ngrams_path=None,
        write_out: bool = False,
        log_samples: bool = True,
        verbosity: str = "INFO",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    # decontaminate = decontamination_ngrams_path is not None

    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)

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
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )

            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[task.OUTPUT_TYPE] += numpad

    # TextParser and Retrievers for Retrieval-Augmented Generation (RAG)
    textParser = TextParser()  # Keyword extractor
    atomicRetriever = AtomicRetriever()
    # gptRetriever = GPTRetriever(api_key="", model_name="gpt-3.5-turbo")
    wikiRetriever = WikiRetriever(full_text=False)
    conceptNetRetriever = ConceptNetRetriever(verbose=False)
    arxivRetriever = ArxivRetriever()
    googleSearchRetriever = GoogleSearchRetriever()

    def get_keywords(_query: str) -> List[str]:
        _keywords_1 = textParser.get_keywords_keybert(_query, n_bag=1)
        _keywords_2 = textParser.get_keywords_keybert(_query, n_bag=2)
        _keywords = sorted(list(set(_keywords_1 + _keywords_2)))
        return _keywords

    # ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            # Evaluator workflow
            #     The evaluator feeds the model with req.arguments[0] (called "context") as input and
            #     compares the model output with req.arguments[0] (called "continuation"), which calculate two values:
            #     1. the logits.sum() of the generation probability;
            #     2. the exact match score (whether the model generation is exactly the same as the reference)
            #     See lm_eval/api/model.py `rem_res = getattr(self.lm, attr)(remaining_reqs)` and `loglikelihood(...)`
            #     See lm_eval/models/huggingface.py `_loglikelihood_tokens(...)` and `multi_logits = F.log_softmax(...)`
            #     Therefore, we need to prepend external knowledge (RAG retrievals) to the front of req.arguments[0]
            # Retrieval-Augmented Generation (RAG) workflow
            #     Step 0: [Optional] Query preprocessing, e.g., rewriting
            #     Step 1: Keywords extraction (using KeyBERT)
            #     Step 2: Search for relevant documents from multiple knowledge bases/sources (using online API)
            #     Step 2.5: [Optional] Document postprocessing, e.g., ranking, refinement, summarization, etc.
            #     Step 3: Augmentation: Combine the documents to the original query (different prompting methods)
            #     Step 3.5: [Optional] Supervised fine-tuning (SFT) / Instruction Tuning / Alignment via RLHF or DPO
            #     Step 4: Run models and get evaluation results
            req.arguments_original = copy.deepcopy(req.arguments)  # Tuple(str): constructed input-output prompt
            # req_docs = req.doc  # dict: task-specific dictionary -> raw attributes to construct our RAG prompt

            # TODO: RAG Step 0: [Optional] Query preprocessing, e.g., rewriting
            # TODO: Different knowledge source API needs different query/keyword input
            task_name = req.task_name
            if task_name == "wsc273":
                # The Winograd Schema Challenge  https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html
                # req.arguments: Tuple(str, str)
                # req.doc: {"text": str, "pronoun": str, "pronoun_loc": int, "quote": str, "quote_loc": int,
                #     "options": List[str], "label": int, "source": str}
                # Task: Pick an option in the `options` list that the `pronoun` at `pronoun_loc` in the `text` refers to
                cur_query = req.doc["text"]
            elif task_name == "winogrande":
                # WinoGrande: Adversarial WSC  https://winogrande.allenai.org/
                # req.arguments: Tuple(str, str)
                # req.doc: {"sentence": str, "option1": str, "option2": str, "answer": str}
                # Task: Pick either `option1` or `option2` to replace the "_" in the `sentence`
                cur_query = ""
            elif task_name == "anli_r1":  # "anli"
                # Adversarial NLI  https://github.com/facebookresearch/anli
                # req.arguments: Tuple(str, str)
                # req.doc: {"uid": str, "premise": str, "hypothesis": str, "label": int, "reason": str}
                # Task: NLI (premise -> hypothesis): label = 0 entailment; label = 1 neutral; label = 2 contradiction
                cur_query = ""
            elif task_name == "anli_r2":  # "anli"
                cur_query = ""
            elif task_name == "anli_r3":  # "anli"
                cur_query = ""
            elif task_name == "arc_easy":  # "ai2_arc"
                # AI2 Reasoning Challenge  https://allenai.org/data/arc
                # req.arguments: Tuple(str, str)
                # req.doc: {"id": str, "question": str, "choices": dict{"text": list, label: list}, "answerKey": str}
                # Task: Answer the question by picking a choice (label) from the choices
                cur_query = ""
            elif task_name == "arc_challenge":  # "ai2_arc"
                cur_query = ""
            elif task_name == "piqa":
                cur_query = ""
            elif task_name == "swag":
                cur_query = ""
            elif task_name == "hellaswag":
                cur_query = ""
            elif task_name == "rte":  # GLUE - "glue"
                cur_query = ""
            elif task_name == "qnli":  # GLUE
                cur_query = ""
            elif task_name == "mnli":  # GLUE
                cur_query = ""
            elif task_name == "mnli_mismatch":  # GLUE
                cur_query = ""
            elif task_name == "mrpc":  # GLUE
                cur_query = ""
            elif task_name == "qqp":  # GLUE
                cur_query = ""
            elif task_name == "wnli":  # GLUE
                cur_query = ""
            elif task_name == "sst2":  # GLUE
                cur_query = ""
            elif task_name == "cola":  # GLUE
                cur_query = ""
            elif task_name == "cb":  # SuperGLUE - "super-glue-lm-eval-v1"
                cur_query = ""
            elif task_name == "wic":  # SuperGLUE
                cur_query = ""
            elif task_name == "sglue_rte":  # SuperGLUE
                cur_query = ""
            elif task_name == "boolq":  # SuperGLUE
                cur_query = ""
            elif task_name == "copa":  # SuperGLUE
                cur_query = ""
            elif task_name == "multirc":  # SuperGLUE
                cur_query = ""
            elif task_name == "record":  # SuperGLUE
                cur_query = ""
            elif task_name == "wsc":  # SuperGLUE
                cur_query = ""
            else:
                raise ValueError(f"ValueError: task_name = {task_name}")

            # TODO: RAG Step 1: Keywords extraction (using KeyBERT)
            cur_keywords = get_keywords(cur_query)

            # TODO: RAG Step 2: Search for relevant documents from multiple knowledge bases/sources (using online API)
            atomic_rag = atomicRetriever.retrieve(cur_query)  # text completion by the Atomic-Comet model
            # gpt_rag = gptRetriever.retrieve(cur_query)  # GPT generation
            wiki_rag = wikiRetriever.retrieve(cur_query)  # wiki pages of the concept
            conceptNet_rag = conceptNetRetriever.retrieve(cur_query)  # all the edges of the concept
            arxiv_rag = arxivRetriever.retrieve(cur_query)  # the Abstract of most relevant N papers
            googleSearch_rag = googleSearchRetriever.retrieve(cur_query)  # Google Search top-N results

            for kw in cur_keywords:
                wiki_rag += wikiRetriever.retrieve(kw)
                conceptNet_rag += conceptNetRetriever.retrieve(kw)

            # TODO: RAG Step 2.5: [Optional] Document postprocessing, e.g., ranking, refinement, summarization, etc.
            # TODO: RAG Step 3: Augmentation: Combine the documents to the original query (different prompting methods)
            rag_context = "RAG:\n"
            rag_context += "Atomic Knowledge:\n" + "\n".join(atomic_rag) + "\n"
            # rag_context += "GPT Knowledge:\n" + "\n".join(gpt_rag) + "\n"
            rag_context += "Wikipedia Knowledge:\n" + "\n".join(wiki_rag) + "\n"
            rag_context += "ConceptNet Knowledge:\n" + "\n".join(conceptNet_rag) + "\n"
            rag_context += "arXiv Knowledge:\n" + "\n".join(arxiv_rag) + "\n"
            rag_context += "Google Search Knowledge:\n" + "\n".join(googleSearch_rag) + "\n"
            req.arguments[0] = rag_context + req.arguments[0]  # Augmentation

            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # TODO: RAG Step 3.5: [Optional] Supervised fine-tuning (SFT) / Instruction Tuning / Alignment via RLHF or DPO
        # Use train.py / train_dp.py / train_ddp.py to fine-tune models; Load the trained model by
        # setting `--model_args "pretrained=/path/to/huggingface_checkpoint,dtype=float" for eval.py

        # TODO: RAG Step 4: Run models and get evaluation results
        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    # ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        # ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
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
        # if multigpu, then gather data across all ranks to rank 0
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
        # ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        results, samples, configs, versions, num_fewshot = consolidate_results(
            eval_tasks
        )

        # ### Calculate group metrics ###
        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if len(task_list) == 0:
                    # task_hierarchy entries are either
                    # `group_name: [subtask1, subtask2, ...]`
                    # or `task_name: []`.
                    # we only want to operate on groups here.
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

                    # gather metrics, sizes, and stderrs from subtasks
                    metrics = [
                        results[task][metric]
                        for task in task_list
                        if metric in results[task]
                    ]  # TODO: copy?
                    stderrs = [
                        results[task][stderr]
                        for task in task_list
                        if stderr in results[task]
                    ]
                    sizes = [
                        results[task]["samples"]
                        for task in task_list
                        if metric in results[task]
                    ]

                    # compute group's pooled metric and stderr
                    results[group][
                        metric
                    ] = lm_eval.api.metrics.aggregate_subtask_metrics(metrics, sizes)
                    # TODO: calculate grouped metric using aggregation fn
                    if "N/A" in stderrs:
                        results[group][stderr] = "N/A"
                    else:
                        results[group][
                            stderr
                        ] = lm_eval.api.metrics.pooled_sample_stderr(stderrs, sizes)
                        # TODO: allow GroupConfigs to choose which variance formula is used, for back-compatibility
                        # To use the old (likely incorrect) variance formula, comment out the above and uncomment this line:
                        # results[group][stderr] = lm_eval.api.metrics.combined_sample_stderr(stderrs, sizes, metrics=metrics)

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
