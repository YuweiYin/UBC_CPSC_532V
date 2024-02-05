# -*- coding:utf-8 -*-

import os
import time
from typing import Tuple

from ConceptNet import ConceptNet
from TextConverter import TextConverter

import spacy
import graphviz
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util


class DijkstraSearch:

    def __init__(self):
        # nlp = spacy.load("en_core_web_sm")
        # nlp = spacy.load("en_core_web_trf")
        nlp = spacy.load("en_core_web_md")
        # nlp = spacy.load("en_core_web_lg")
        self.nlp = nlp
        self.kw_model = KeyBERT()
        self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # self.simi_dict = dict()  # cache for the computed similarities (key: a word; value: similarity to tgt_word)
        self.src_word = ""
        self.tgt_word = ""
        self.src_emb = None
        self.tgt_emb = None
        self.similarity_limit = 0.3  # if the similarity between the next node and target one, ignore this node

    def word_similarity(self, word_1: str, word_2: str) -> float:
        string = f"{word_1.strip()} {word_2.strip()}"
        tokens = self.nlp(string)
        similarity = tokens[0].similarity(tokens[1])
        return similarity

    def dijkstra_path_search(self, src_word: str, tgt_word: str, max_depth: int = 10, verbose: bool = True):
        if verbose:
            print(f">>> dijkstra_path_search (single-source weighted BFS): "
                  f"From {src_word} To {tgt_word}; max_depth = {max_depth}")
        # self.simi_dict = dict()  # cache for the computed similarities (key: a word; value: similarity to tgt_word)

        src_word = src_word.strip()
        tgt_word = tgt_word.strip()
        self.src_word = src_word
        self.tgt_word = tgt_word
        self.src_emb = self.st_model.encode(src_word, convert_to_tensor=True)
        self.tgt_emb = self.st_model.encode(tgt_word, convert_to_tensor=True)
        # src_tgt_similarity_word = self.word_similarity(src_word, tgt_word)
        src_tgt_similarity_sent = float(util.pytorch_cos_sim(self.src_emb, self.tgt_emb))
        if verbose:
            # print(f"Src-Tgt Similarity (word level): {src_tgt_similarity_word:.3f}")
            print(f"Similarity between {src_word} and {tgt_word}: {src_tgt_similarity_sent:.3f}")
        # if src_word not in self.simi_dict:
        #     self.simi_dict[src_word] = src_tgt_similarity_sent

        # self.similarity_limit = src_tgt_similarity_sent / 2
        self.similarity_limit = src_tgt_similarity_sent * 2 / 3

        bfs = []  # the set of concept ids
        visited = set()  # avoid looping
        weights = dict()  # edge weight in Dij alg
        prev_node = dict()  # to backtrace: prev_node[next_node] = cur_node
        relations = dict()  # to backtrace: relations[next_node] = [relation, direction]
        matches = list()  # matched nodes (~= tgt_word)
        matches_set = set()  # set(matches)

        continue_bfs_flag = True

        conceptNet = ConceptNet()  # ConceptNet toolkit

        # Initialization
        src_node = conceptNet.get_concept_id(src_word)  # word to concept_id (node id)
        bfs.append(src_node)
        weights[src_node] = 0
        prev_node[src_node] = None  # root node
        relations[src_node] = None

        for cur_depth in range(max_depth):
            if not continue_bfs_flag:
                break
            timer_s = time.perf_counter()
            # if verbose:
            #     print(f">>> >>> cur_depth: {cur_depth}; len(bfs): {len(bfs)}")
            if len(bfs) == 0:
                break

            ignore_node_cnt = 0
            # reused_simi_cnt = 0

            next_bfs = []  # the nodes of the next depth
            for cur_node in bfs:  # deal with all the nodes of the current depth
                # Get the current node (concept id)
                if cur_node not in visited:  # avoid visiting the same node and forming a loop in prev_node
                    visited.add(cur_node)
                else:
                    continue
                # if verbose:
                #     print(f"{cur_node}; cur_depth: {cur_depth}; len(bfs): {len(bfs)}")
                assert isinstance(cur_node, str) and cur_node.startswith(conceptNet.prefix_cid)
                if cur_node in matches_set:
                    continue

                # Match the current node with the target word
                match_result = self.match(tgt_word, cur_node)
                if match_result > 0:  # match the target word
                    matches_set.add(cur_node)
                    matches.append(cur_node)
                    if verbose:
                        # print(f"We are thrilled to announce the {len(matches_set)}th matched node: {cur_node}!\n")
                        print(f">>> *** Number {len(matches_set)} matched node: {cur_node}")
                        # self.print_path(cur_node, prev_node, weights)
                        # self.path_visualization_by_prev(cur_node, prev_node, weights, tgt_word)
                    # if match_result == 1:  # exact match a word in the same type/category of the target word
                    #     continue_bfs_flag = False  # end the whole Dij search process
                    #     break
                    # continue  # don't go further after matching the current node
                    # Once matching a word in the same type/category as the target word, end the Dij process
                    continue_bfs_flag = False  # end the whole Dij search process
                    break

                # Get the corresponding concept from ConceptNet
                cur_url = conceptNet.get_url(cur_node)
                concept = conceptNet.get_concept(cur_url, verbose=verbose)
                if not isinstance(concept, dict) or "edges" not in concept:
                    continue

                # Dealing with its neighbors (related concepts)
                edges = concept["edges"]
                if not isinstance(edges, list) or len(edges) == 0:
                    continue
                for edge in edges:
                    assert isinstance(edge, dict)
                    next_node, weight, relation, direction = conceptNet.get_next_node(cur_node, edge)
                    if next_node in visited or not next_node.startswith(conceptNet.prefix_cid):
                        continue

                    # Optimize by ignoring irrelevant nodes based on similarity
                    next_node_name = next_node.split("/")[-1].replace("_", " ")
                    # if next_node_name in self.simi_dict:  # cache the computed similarities to speed up
                    #     cur_similarity_sent = self.simi_dict[next_node_name]
                    #     reused_simi_cnt += 1
                    # else:
                    #     next_emb = self.st_model.encode(next_node_name, convert_to_tensor=True)
                    #     cur_similarity_sent = float(util.pytorch_cos_sim(next_emb, self.tgt_emb))
                    next_emb = self.st_model.encode(next_node_name, convert_to_tensor=True)
                    cur_similarity_sent = float(util.pytorch_cos_sim(next_emb, self.tgt_emb))
                    if cur_similarity_sent < self.similarity_limit:
                        ignore_node_cnt += 1
                        continue

                    # Add the new node to the path, update the weight.
                    if next_node not in weights:
                        assert next_node not in prev_node, "Node hasn't been visited."
                        next_bfs.append(next_node)  # next_node will be dealt in the next depth
                        weights[next_node] = weights[cur_node] + weight
                        prev_node[next_node] = cur_node
                        relations[next_node] = [relation, direction]
                    # If the weight of a new path is greater than the path before, replace the most likely path.
                    elif weights[next_node] < weights[cur_node] + weight:
                        assert next_node in prev_node, "Node has been visited."
                        weights[next_node] = weights[cur_node] + weight
                        prev_node[next_node] = cur_node
                        relations[next_node] = [relation, direction]

            timer_e = time.perf_counter()
            timer_sec, timer_min = timer_e - timer_s, (timer_e - timer_s) / 60
            if verbose:
                print(f">>> >>> Depth [{cur_depth + 1}]: BFS source nodes {len(bfs)}; "
                      f"Visited neighbors {len(next_bfs)} (ignored {ignore_node_cnt}); "
                      f"Running time {timer_sec:.1f} sec ({timer_min:.1f} min)")
            bfs = next_bfs  # next depth

        # Return the path list
        path_list = []
        if len(matches) == 0:
            print(f"Path between {src_word} and {tgt_word} does not exist or is too long (> {max_depth}).\n")
        for matched_node in matches:
            path_list.append(self.get_path(matched_node, prev_node, weights, relations))
        return path_list

    @staticmethod
    def match(tgt_word: str, node: str) -> int:
        # string preproc
        tgt_word = tgt_word.strip().replace("_", " ")
        node_words = node.split("/")
        node_words = [w.strip().replace("_", " ") for w in node_words]

        # match the main concept
        if len(node_words) >= 4:  # e.g., "/c/en/food"
            if tgt_word == node_words[3]:
                return 1  # exact match the target word

        # match the sub-concept
        if len(node_words) >= 7:  # e.g., "/c/en/butter/n/wn/food"
            if tgt_word == node_words[6]:
                return 2  # exact match a word in the same type/category of the target word

        # for word in node_words:
        #     # if tgt_word in part:
        #     if tgt_word == word:
        #         return True
        return 0  # not matched

        # node_word = node.split("/")[-1]
        # return tgt_word == node_word

    @staticmethod
    def get_path(node: str, prev_node: dict, weights: dict, relations: dict,
                 do_print: bool = False) -> Tuple[list, list, list]:
        node_list = []
        w_list = []
        r_list = []
        cur_node = node
        while isinstance(cur_node, str):
            node_list.append(cur_node)
            cur_r = relations[cur_node]
            r_list.append(cur_r)
            cur_w = weights[cur_node]
            w_list.append(cur_w)
            if do_print:
                print(f"{node}, with a weight of {cur_w}.")
                if cur_r[1] == 0:
                    print(f"Backward, relationship is {cur_r[0]}.")
                else:
                    print(f"Forward, relationship is {cur_r[0]}")
            cur_node = prev_node[cur_node]

        assert len(node_list) == len(w_list)
        # path_len = len(node_list)
        # w_sum = sum(w_list)
        # w_avg = float(w_sum / path_len) if path_len > 0 else 0.0
        # print(f"Path Length: {path_len}; Weight Sum: {w_sum}; Average Weight: {w_avg}\n")
        return node_list, w_list, r_list

    def print_path(self, node: str, prev_node: dict, weights: dict, relations: dict) -> None:
        print(f"Valid Path:")
        node_list, w_list, r_list = self.get_path(node, prev_node, weights, relations, do_print=True)
        print(f"Path ends. Path Length: {len(w_list)}; Weight Sum: {sum(w_list)}\n")

    def path_visualization_by_list(self, node_list: list, w_list: list, r_list: list,
                                   e_id: str, c_id: str, save_dir: str = "output_figure") -> Tuple[int, float]:
        assert len(node_list) == len(w_list)
        path_len = len(node_list)
        w_sum = sum(w_list)
        w_avg = float(w_sum / path_len) if path_len > 0 else 0.0
        # print(f"Path Length: {path_len}; Weight Sum: {w_sum}; Average Weight: {w_avg}\n")

        # Draw a figure of the path
        src_name = self.src_word.replace("/", "_")
        tgt_name = self.tgt_word.replace("/", "_")
        dot = graphviz.Digraph(name=f"{e_id}_{c_id}---{src_name}-{tgt_name}---"
                                    f"PLen{path_len}_WSum{w_sum:.3f}_WAvg{w_avg:.3f}",
                               comment=f"The path the concepts from '{src_name}' to '{tgt_name}'."
                                       f"Path Length: {path_len}; Weight Sum: {w_sum}; Average Weight: {w_avg}")

        for idx, cur_node in enumerate(node_list):
            # dot.node(name=str(idx), label=str(cur_node), _attributes={"weight": f"{w_list[idx]:.3f}"})
            # dot.node(name=str(idx), label=str(cur_node), _attributes={})
            dot.node(name=str(idx), label=TextConverter.extract_word(str(cur_node)), _attributes={})

        for idx in range(len(node_list) - 1):  # reversely link nodes
            # dot.edge(tail_name=str(idx), head_name=str(idx + 1), _attributes={"weight": f"{w_list[idx]:.3f}"})
            direction = "back" if r_list[idx][1] == 1 else "forward"
            # dot.edge(tail_name=str(idx), head_name=str(idx + 1), label=r_list[idx][0],
            #          _attributes={"weight": f"{w_list[idx]:.3f}", "dir": direction})
            dot.edge(tail_name=str(idx), head_name=str(idx + 1), label=f"{r_list[idx][0]}; {w_list[idx]:.3f}",
                     _attributes={"weight": f"{w_list[idx]:.3f}", "dir": direction})

        # print(dot.source)

        """
        # For macOS, if the following Exception occurs, run the bash script `brew install graphviz`
        # >>> raise ExecutableNotFound(cmd) from e
        #     graphviz.backend.execute.ExecutableNotFound: failed to execute PosixPath('dot'),
        #     make sure the Graphviz executables are on your systems' PATH
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        dot.render(directory=save_dir, format="pdf").replace("\\", "/")
        return path_len, w_sum

    def path_visualization_by_prev(self, node: str, prev_node: dict, weights: dict, relations: dict,
                                   e_id: str, c_id: str) -> Tuple[int, float]:
        # Get the path
        node_list, w_list, r_list = self.get_path(node, prev_node, weights, relations, do_print=False)
        assert len(node_list) == len(w_list)

        # Draw a figure of the path
        path_len, w_sum = self.path_visualization_by_list(node_list, w_list, r_list, e_id, c_id)
        return path_len, w_sum
