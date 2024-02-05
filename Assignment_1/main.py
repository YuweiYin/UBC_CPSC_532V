#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import time
import json
import argparse

# from DijkstraSearch import DijkstraSearch
from DijkstraSearchBiSource import DijkstraSearchBiSource
from TextParser import TextParser
from TextConverter import TextConverter


def run() -> None:
    textParser = TextParser()

    res_list = []
    acc_list = []
    ans_list = []
    pre_list = []

    ord_a = ord("a")
    idx2option = {index: chr(ord_a + index) for index in range(26)}

    from data.examples_dict import examples

    max_depth = int(args.max_depth)
    target_e_id = int(args.example_id)
    # if target_e_id >= 0:
    #     assert target_e_id < len(examples)
    #     examples = [examples[target_e_id]]

    for example in examples:
        timer_start_example = time.perf_counter()

        e_id = example["id"]
        if 0 <= target_e_id != e_id:
            continue

        context = example["context"]
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]

        res_dict = dict()
        res_dict["example_id"] = e_id
        res_dict["max_depth"] = max_depth
        res_dict["context"] = context
        res_dict["question"] = question
        res_dict["choices"] = choices
        # res_dict["answer"] = answer

        question = context + question
        question = question.strip()
        choices = [c.strip() for c in choices]

        target_c_id = int(args.choice_id)
        if target_c_id >= 0:
            assert target_c_id < len(choices)
            examples = [choices[target_c_id]]

        # Parse and filter questions and choices
        max_number_kw = 5
        n_bag = 1
        q_keywords = textParser.get_keywords_keybert(question, n_bag=n_bag)  # get keywords
        q_keywords = list(set(q_keywords))  # remove duplication
        c_kw_list = []
        for choice in choices:
            c_keywords = textParser.get_keywords_keybert(choice, n_bag=n_bag)  # get keywords
            c_keywords = list(set(c_keywords))  # remove duplication
            # sort c_keywords by computing the average similarity between the current c_kw and each q_kw in q_keywords
            c_kw_sorted = textParser.keyword_sort(references=q_keywords, keywords=c_keywords)
            c_kw_sorted = c_kw_sorted[:max_number_kw]
            c_kw_list.append(c_kw_sorted)
        q_keywords = q_keywords[:max_number_kw]

        # get rid of the common kw in all c_keywords
        common_kw_set = set(c_kw_list[0])
        for c_keywords in c_kw_list[1:]:
            common_kw_set = common_kw_set & set(c_keywords)
        c_keywords_list = []
        if len(common_kw_set) > 0:
            for c_kw in c_kw_list:
                c_kw = list(set(c_kw) - common_kw_set)
                c_keywords_list.append(c_kw)
        else:
            c_keywords_list = c_kw_list

        res_dict["q_keywords"] = q_keywords
        res_dict["c_keywords_list"] = c_keywords_list
        assert len(c_keywords_list) == len(choices)

        choice_score_final = []

        for c_id, c_keywords in enumerate(c_keywords_list):
            cur_choice_id = idx2option[c_id]
            cur_choice_text = choices[c_id]
            # Dijkstra path search
            # dij = DijkstraSearch()  # single-source weighted BFS
            dij = DijkstraSearchBiSource()  # bi-source weighted BFS
            total_pair_n = 0  # denominator
            match_pair_n = 0  # numerator
            pair_score = []  # best_path and best_config of each Dij search

            for src_word in c_keywords:
                for tgt_word in q_keywords:
                    if src_word == tgt_word:
                        continue

                    print(f"\n\nExample: {e_id}; Choice: \"{cur_choice_text}\"; From \"{src_word}\" To \"{tgt_word}\"")
                    total_pair_n += 1

                    path_list = dij.dijkstra_path_search(src_word, tgt_word, verbose=True, max_depth=max_depth)
                    if len(path_list) == 0:
                        # choice_score.append(0.0)  # no matched nodes -> score 0
                        continue
                    match_pair_n += 1

                    best_path = [[], [], []]  # For the current pair, the best path of all returned paths
                    best_config = [0.0, 0.0, 0.0]  # the best path_len, w_sun, w_avg
                    for path in path_list:
                        node_list, w_list, r_list = path
                        assert len(node_list) == len(w_list) + 1
                        path_len = len(w_list)
                        w_sum = sum(w_list)
                        w_avg = float(w_sum / path_len) if path_len > 0 else 0.0
                        if w_avg > best_config[2]:  # update the best path
                            best_path = path
                            best_config = [path_len, w_sum, w_avg]

                    pair_score.append((best_path, best_config))

                    # Visualize the best path
                    dij.path_visualization_by_list(
                        best_path[0], best_path[1], best_path[2], str(e_id), idx2option[c_id])

                    # Present the information in natural language.
                    #   Convert the relations to natural language. For example, "Flu is a type of influenza.
                    #   Influenza is a type of disease. Disease and illness have similar meanings.
                    #   Staying in bed is used for illness. Staying in bed requires a bed.
                    #   Bed is located in a house. House is a type of home. Home is the opposite of party."
                    #   See [this](https://aclanthology.org/D19-1109/) for ConceptNet relation templates.
                    #   This function will also be used in the next assignment.

                    textConverter = TextConverter(best_path, bool(args.use_gpt))
                    textConverter.convert(str(e_id), idx2option[c_id])
                    # try:
                    #     textConverter = TextConverter(best_path)
                    #     # textConverter.convert(str(e_id), idx2option[c_id], use_gpt=True)
                    #     textConverter.convert(str(e_id), idx2option[c_id], use_gpt=False)
                    # except Exception as e:
                    #     print(">>> >>> TextConverter unsuccessful. "
                    #           "Possibly because the OPENAI_API_KEY is incorrect or expired.")
                    #     print(e)
                    #     continue

            # Compute the score of the current choice
            #   use the avg_w (average edge weight in ConceptNet) and match ratio as the metric to choose an answer
            #   For example, there are 5 q_keywords and 4 c_keywords for a choice, the number of search trials is 20
            #   if 15 of 20 keyword pairs matched in max_depth, i.e., match_pair_n == 15 and total_pair_n == 20,
            #   then the score of the current choice is as follows:
            #   $\frac{15}{20} \times avg_w$, where avg_w = sum(edge_w) / #edges of all best paths in all Dij search
            #   if src_word == tgt_word, skip this pair (This will affect both the denominator and numerator)
            total_path_len, total_w_sum, total_w_avg = 0.0, 0.0, 0.0
            for best_path, best_config in pair_score:
                total_path_len += best_config[0]
                total_w_sum += best_config[1]
                total_w_avg += best_config[2]
            if total_path_len > 0:  # micro average
                cur_choice_score = float((total_w_sum / total_path_len) * (match_pair_n / total_pair_n))
            else:
                cur_choice_score = 0.0
            print(f"\n*** Example: {e_id}; Choice: {cur_choice_id}; SCORE: {cur_choice_score:.3f} "
                  f"(total_w_sum = {total_w_sum:.3f}; total_path_len = {int(total_path_len)}; "
                  f"match_pair_n = {match_pair_n}; total_pair_n = {total_pair_n})")
            choice_score_final.append(cur_choice_score)

        # Rank the choices, and then pick a choice or multiple choices
        score_option = []
        for idx, score_final in enumerate(choice_score_final):
            score_option.append((score_final, idx2option[idx]))
        score_option_sort = sorted(score_option, key=lambda x: x[0], reverse=True)  # ranking

        predict = [so[1] for so in score_option_sort[:len(answer)]]
        answer.sort()
        predict.sort()
        acc = 1 if answer == predict else 0
        ans_list.append(answer)
        pre_list.append(predict)
        acc_list.append(acc)

        res_dict["answer"] = answer
        res_dict["predict"] = predict

        timer_end_example = time.perf_counter()
        time_sec, time_min = timer_end_example - timer_start_example, (timer_end_example - timer_start_example) / 60
        print(f"\n*** DONE *** Example: {e_id} - Running Time: {time_sec:.1f} sec ({time_min:.1f} min)")
        res_dict["time_sec"] = time_sec
        res_dict["time_min"] = time_min

        res_list.append(res_dict)

    print(f"\nAccuracy: {sum(acc_list) / len(acc_list)}.\nPredictions: {pre_list}\nAnswers: {ans_list}")

    res_dir = os.path.join("./output/")
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    if target_e_id > 0:
        res_fp = os.path.join(res_dir, f"eval_result_example{target_e_id}.jsonl")
    else:
        res_fp = os.path.join(res_dir, "eval_result_all.jsonl")
    with open(res_fp, "w", encoding="utf-8") as fp_out:
        for res in res_list:
            line = json.dumps(res) + "\n"
            fp_out.write(line)
    print("\nDone!")


def stat() -> None:
    """
    Statistics of the path length, total weights, avg weight, and relation types
    """

    path_length = []
    path_weight_sum = []
    path_weight_avg = []
    relation_counter = dict()

    in_dir = "./output_text/"
    assert os.path.isdir(in_dir)
    fn_list = os.listdir(in_dir)
    fp_list = [os.path.join(in_dir, fn) for fn in fn_list if fn.endswith(".json")]

    for fp in fp_list:
        if not os.path.isfile(fp):
            print(f"PATH ERROR: fp = {fp}")
            continue
        with open(fp, "r", encoding="utf-8") as fp_in:
            cur_path = json.load(fp_in)

        # e_id = cur_path["example_id"]
        # c_id = cur_path["choice_id"]
        # start_node = cur_path["start_node"]
        # end_node = cur_path["end_node"]
        # node_list = cur_path["node_list"]
        w_list = cur_path["w_list"]
        r_list = cur_path["r_list"]
        # node_relation_list = cur_path["node_relation_list"]
        # prompt_list = cur_path["prompt_list"]
        # description_list = cur_path["description_list"]

        p_len = len(r_list)
        p_weight_sum = sum(w_list)
        p_weight_avg = float(p_weight_sum / p_len) if p_len > 0 else 0.0

        path_length.append(p_len)
        path_weight_sum.append(p_weight_sum)
        path_weight_avg.append(p_weight_avg)

        for r in r_list:
            rel, direction = r
            if rel not in relation_counter:
                relation_counter[rel] = 1
            else:
                relation_counter[rel] += 1

    rel_cnt = list(relation_counter.items())
    rel_cnt_sorted = sorted(rel_cnt, key=lambda x: x[1], reverse=True)

    stat_result = {
        "avg_path_length": sum(path_length) / len(path_length) if len(path_length) > 0 else 0,
        "avg_path_weight_sum": sum(path_weight_sum) / len(path_weight_sum) if len(path_weight_sum) > 0 else 0,
        "avg_path_weight_avg": sum(path_weight_avg) / len(path_weight_avg) if len(path_weight_avg) > 0 else 0,
        "top_10_relation": rel_cnt_sorted[:10],
        "path_length": path_length,
        "path_weight_sum": path_weight_sum,
        "path_weight_avg": path_weight_avg,
        "relation_counter": rel_cnt_sorted,
    }
    res_dir = os.path.join("./output/")
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    res_fp = os.path.join(res_dir, "stat_results.json")
    with open(res_fp, "w", encoding="utf-8") as fp_out:
        json.dump(stat_result, fp_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-e", "--example_id", type=int, default=-1, help="-1: all; 0: examples[0]")
    parser.add_argument("-c", "--choice_id", type=int, default=-1, help="-1: all; 0: choices[0]")
    parser.add_argument("-d", "--max_depth", type=int, default=6, help="Max BFS depth")
    parser.add_argument("-t", "--use_gpt", action="store_true", default=False, help='Use LLM/GPT')
    args = parser.parse_args()
    print(args)

    timer_start = time.perf_counter()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    run()
    # stat()

    timer_end = time.perf_counter()
    print("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    sys.exit(0)
