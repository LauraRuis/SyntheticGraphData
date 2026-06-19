#!/usr/bin/env python3
"""Small, standalone graph data generator.

The script writes three JSONL files:

    train.jsonl
    val.jsonl
    test.jsonl

Every row has the same keys:

    prompt, completion, answer, split, kind, primary_inference_type,
    inference_types, explanation_type, query_entity, observed_entities,
    observed_values, graph_idx

Graph drawing
uses matplotlib and networkx if ``--visualize`` is set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


TRUE_LABEL = "1"
FALSE_LABEL = "0"


def boxed(answer: str) -> str:
    """Return the answer in the format expected by the model."""
    return f"\\boxed{{{answer}}}"


def all_entity_names() -> list[str]:
    """All short, readable entity names: AA, AB, AC, ..."""
    return [
        chr(a) + chr(b)
        for a in range(ord("A"), ord("Z") + 1)
        for b in range(ord("A"), ord("Z") + 1)
    ]


def make_entity_names(num_entities: int) -> list[str]:
    """Make short, readable entity names: AA, AB, AC, ..."""
    names = all_entity_names()
    if num_entities > len(names):
        raise ValueError(f"Can only make {len(names)} two-letter entity names.")
    return names[:num_entities]


def split_items(items: list, val_ratio: float, test_ratio: float, rng: random.Random):
    """Randomly split items into train/val/test lists.

    We keep this deliberately simple.  If there are at least three items and a
    heldout ratio is positive, val/test each get at least one item.
    """
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = round(n * val_ratio)
    n_test = round(n * test_ratio)
    if n >= 3 and val_ratio > 0:
        n_val = max(1, n_val)
    if n >= 3 and test_ratio > 0:
        n_test = max(1, n_test)

    # Prefer to leave at least one training item when possible.
    while n_val + n_test > max(0, n - 1):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test


def split_name_for_mentioned_entities(mentioned: set[str], entity_to_split: dict[str, str]) -> str:
    """Choose which split an explanation belongs to.

    If an explanation mentions a heldout val/test entity, it goes to that
    heldout split.  Otherwise it is a training explanation.
    """
    splits = {entity_to_split[e] for e in mentioned if e in entity_to_split}
    if "test" in splits:
        return "test"
    if "val" in splits:
        return "val"
    return "train"


class SimpleGraphDataset:
    """A small graph dataset with observations and heldout explanations.

    Each graph is a random DAG.  Root nodes are sampled randomly in each world.
    Non-root nodes are deterministic:

      OR node  = true if any parent is true
      AND node = true if all parents are true
    """

    INFERENCE_PRIORITY = ("ancestor", "descendant", "independent")

    RULE_QUESTIONS = [
        "What is the rule that determines {E}? Express the rule in the format \\boxed{{answer}}.",
        "How is {E} computed from its inputs? Express the rule in the format \\boxed{{answer}}.",
        "Describe the logical rule for {E}. Express the rule in the format \\boxed{{answer}}.",
        "What determines whether {E} is True or False? Express the rule in the format \\boxed{{answer}}.",
        "Explain how {E} depends on other entities. Express the rule in the format \\boxed{{answer}}.",
        "State the condition under which {E} becomes True. Express the rule in the format \\boxed{{answer}}.",
        "What logical expression defines {E}? Express the rule in the format \\boxed{{answer}}.",
        "Write the update rule for {E}. Express the rule in the format \\boxed{{answer}}.",
        "How do the direct inputs combine to set {E}? Express the rule in the format \\boxed{{answer}}.",
        "What input pattern makes {E} true or false? Express the rule in the format \\boxed{{answer}}.",
    ]
    PARENT_RETRIEVAL_QUESTIONS = [
        "What are the direct parents of {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Which entities directly determine {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Name the entities that are direct inputs to {E}. Give the entity names in the format \\boxed{{answer}}.",
        "{E} is directly determined by which entities? Give the entity names in the format \\boxed{{answer}}.",
        "List the direct parents of {E}. Give the entity names in the format \\boxed{{answer}}.",
        "Which nodes feed directly into {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Identify the immediate predecessors of {E}. Give the entity names in the format \\boxed{{answer}}.",
        "Which entities are one edge upstream of {E}? Give the entity names in the format \\boxed{{answer}}.",
        "What entities act as the immediate causes of {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Which entities connect directly into {E}? Give the entity names in the format \\boxed{{answer}}.",
    ]
    CHILD_RETRIEVAL_QUESTIONS = [
        "What are the direct children of {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Which entities does {E} directly influence? Give the entity names in the format \\boxed{{answer}}.",
        "Name the entities that {E} is a direct input to. Give the entity names in the format \\boxed{{answer}}.",
        "{E} directly determines which entities? Give the entity names in the format \\boxed{{answer}}.",
        "List the direct children of {E}. Give the entity names in the format \\boxed{{answer}}.",
        "Which nodes are directly downstream of {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Identify the immediate successors of {E}. Give the entity names in the format \\boxed{{answer}}.",
        "Which entities receive a direct edge from {E}? Give the entity names in the format \\boxed{{answer}}.",
        "What entities are immediately affected by {E}? Give the entity names in the format \\boxed{{answer}}.",
        "Which entities are connected directly out of {E}? Give the entity names in the format \\boxed{{answer}}.",
    ]
    RELATIONSHIP_QUESTIONS = [
        "What is the relationship between {P1} and {P2}? Express the relationship in the format \\boxed{{answer}}.",
        "How are {P1} and {P2} related? Express the relationship in the format \\boxed{{answer}}.",
        "Describe the relationship between {P1} and {P2}. Express the relationship in the format \\boxed{{answer}}.",
        "What dependency exists between {P1} and {P2}? Express the relationship in the format \\boxed{{answer}}.",
        "Characterize the connection between {P1} and {P2}. Express the relationship in the format \\boxed{{answer}}.",
        "What kind of dependency links {P1} and {P2}? Express the relationship in the format \\boxed{{answer}}.",
        "What graph relation holds between {P1} and {P2}? Express the relationship in the format \\boxed{{answer}}.",
        "State the relationship between {P1} and {P2}. Express the relationship in the format \\boxed{{answer}}.",
        "Describe how {P1} and {P2} are connected in the graph. Express the relationship in the format \\boxed{{answer}}.",
        "What does knowing {P1} tell you about {P2}? Express the relationship in the format \\boxed{{answer}}.",
    ]

    def __init__(
        self,
        num_graphs: int = 10,
        num_entities: int = 20,
        edge_prob: float = 0.2,
        node_or_prob: float = 1.0,
        num_worlds: int = 1000,
        sample_ratio_from_worlds: float = 0.01,
        root_prior: float = 0.5,
        max_obs_size: int = 2,
        num_observation_templates: int = 10,
        num_explanation_templates: int = 10,
        max_observations: int | None = None,
        max_explanations: int | None = None,
        num_copies_explanations: int = 4,
        add_instruction_datapoints: int = 2500,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        heldout_scope: str = "nodes",
        seed: int = 0,
    ):
        self.num_graphs = num_graphs
        self.num_entities = num_entities
        self.edge_prob = edge_prob
        self.node_or_prob = node_or_prob
        self.num_worlds = num_worlds
        if not 0.0 <= sample_ratio_from_worlds <= 1.0:
            raise ValueError(
                f"sample_ratio_from_worlds must be between 0 and 1, got {sample_ratio_from_worlds}"
            )
        self.sample_ratio_from_worlds = sample_ratio_from_worlds
        self.root_prior = root_prior
        self.max_obs_size = max_obs_size
        self.num_observation_templates = num_observation_templates
        self.num_explanation_templates = num_explanation_templates
        if max_observations is not None and max_observations < 0:
            raise ValueError(
                f"max_observations must be non-negative or None, got {max_observations}"
            )
        if max_explanations is not None and max_explanations < 0:
            raise ValueError(
                f"max_explanations must be non-negative or None, got {max_explanations}"
            )
        if num_copies_explanations < 0:
            raise ValueError(
                f"num_copies_explanations must be non-negative, got {num_copies_explanations}"
            )
        if add_instruction_datapoints < 0:
            raise ValueError(
                f"add_instruction_datapoints must be non-negative, got {add_instruction_datapoints}"
            )
        self.max_observations = max_observations
        self.max_explanations = max_explanations
        self.num_copies_explanations = num_copies_explanations
        self.add_instruction_datapoints = add_instruction_datapoints
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.heldout_scope = heldout_scope
        self.seed = seed
        self.rng = random.Random(seed)

        self.graphs = []
        self.examples = []
        self._make_graphs()
        self._make_explanation_split()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _make_graphs(self) -> None:
        """Create random DAGs and node rules."""
        available_names = all_entity_names()
        total_needed = self.num_graphs * self.num_entities
        if total_needed > len(available_names):
            raise ValueError(
                f"num_graphs * num_entities = {total_needed} exceeds "
                f"{len(available_names)} available AA-ZZ entity names."
            )

        for graph_idx in range(self.num_graphs):
            entity_rng = random.Random(self.seed + graph_idx)
            entities = entity_rng.sample(available_names, self.num_entities)
            chosen = set(entities)
            available_names = [name for name in available_names if name not in chosen]

            # A random topological order makes it easy to avoid cycles: only
            # add edges from earlier nodes to later nodes.
            order = list(entities)
            self.rng.shuffle(order)
            order_index = {entity: i for i, entity in enumerate(order)}

            children = {entity: set() for entity in entities}
            parents = {entity: set() for entity in entities}
            for i, parent in enumerate(order):
                for child in order[i + 1 :]:
                    if self.rng.random() < self.edge_prob:
                        children[parent].add(child)
                        parents[child].add(parent)

            rules = {}
            for entity in entities:
                if not parents[entity]:
                    rules[entity] = "ROOT"
                elif self.rng.random() < self.node_or_prob:
                    rules[entity] = "OR"
                else:
                    rules[entity] = "AND"

            self.graphs.append(
                {
                    "graph_idx": graph_idx,
                    "entities": entities,
                    "order": order,
                    "order_index": order_index,
                    "children": children,
                    "parents": parents,
                    "rules": rules,
                }
            )

    def _make_explanation_split(self) -> None:
        """Decide which explanations are train/val/test.

        Observations are always training data.  Only explanations are held out.

        --heldout-scope graphs:
            whole graph structures are assigned to train/val/test.

        --heldout-scope nodes:
            individual nodes are assigned to train/val/test.
        """
        self.entity_to_explanation_split = {}
        if self.heldout_scope == "graphs":
            graph_indices = [g["graph_idx"] for g in self.graphs]
            train_graphs, val_graphs, test_graphs = split_items(
                graph_indices, self.val_ratio, self.test_ratio, self.rng
            )
            graph_to_split = {
                **{idx: "train" for idx in train_graphs},
                **{idx: "val" for idx in val_graphs},
                **{idx: "test" for idx in test_graphs},
            }
            for graph in self.graphs:
                split = graph_to_split[graph["graph_idx"]]
                for entity in graph["entities"]:
                    self.entity_to_explanation_split[entity] = split
        elif self.heldout_scope == "nodes":
            for graph in self.graphs:
                train_nodes, val_nodes, test_nodes = split_items(
                    graph["entities"], self.val_ratio, self.test_ratio, self.rng
                )
                for entity in train_nodes:
                    self.entity_to_explanation_split[entity] = "train"
                for entity in val_nodes:
                    self.entity_to_explanation_split[entity] = "val"
                for entity in test_nodes:
                    self.entity_to_explanation_split[entity] = "test"
        else:
            raise ValueError("--heldout-scope must be either 'graphs' or 'nodes'.")

    # ------------------------------------------------------------------
    # World sampling
    # ------------------------------------------------------------------

    def sample_world(self, graph: dict) -> dict[str, bool]:
        """Sample one world by random roots and deterministic propagation."""
        world = {}
        queue = [entity for entity in sorted(graph["entities"]) if not graph["parents"][entity]]

        for root in queue:
            world[root] = self.rng.random() < self.root_prior

        processed = set(queue)
        idx = 0
        while idx < len(queue):
            entity = queue[idx]
            idx += 1
            for child in sorted(graph["children"][entity]):
                if child in processed:
                    continue
                parents = graph["parents"][child]
                if not all(parent in processed for parent in parents):
                    continue
                rule = graph["rules"][child]
                if rule == "OR":
                    world[child] = any(world[parent] for parent in parents)
                elif rule == "AND":
                    world[child] = all(world[parent] for parent in parents)
                else:
                    raise ValueError(f"Unknown rule: {rule}")
                processed.add(child)
                queue.append(child)
        return world

    # ------------------------------------------------------------------
    # Graph reasoning helpers
    # ------------------------------------------------------------------

    def ancestors(self, graph: dict, entity: str) -> set[str]:
        """All nodes with a directed path into entity."""
        found = set()
        stack = list(graph["parents"][entity])
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(graph["parents"][node])
        return found

    def descendants(self, graph: dict, entity: str) -> set[str]:
        """All nodes reachable by following directed edges out of entity."""
        found = set()
        stack = list(graph["children"][entity])
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(graph["children"][node])
        return found

    def relation_type(self, graph: dict, first: str, second: str) -> str:
        """Return ancestor, descendant, or independent for an ordered pair."""
        if second in self.descendants(graph, first):
            return "ancestor"
        if first in self.descendants(graph, second):
            return "descendant"
        return "independent"

    def classify_inference_types(self, graph: dict, observed_entities: list[str], query_entity: str) -> list[str]:
        """Simple structural relation between observations and the query.

        For an observation O and query Q:
          ancestor    means O has a directed path to Q
          descendant  means Q has a directed path to O
          independent means neither direction has a directed path

        With several observations, a single example can have several relation
        types.  We keep all of them and let primary_inference_type use the
        priority below.
        """
        relation_types = {
            self.relation_type(graph, observed_entity, query_entity)
            for observed_entity in observed_entities
        }
        return [
            label for label in self.INFERENCE_PRIORITY
            if label in relation_types
        ] or ["independent"]

    def classify_inference(self, graph: dict, observed_entities: list[str], query_entity: str) -> str:
        """Return the primary observation inference type."""
        return self.classify_inference_types(graph, observed_entities, query_entity)[0]

    # ------------------------------------------------------------------
    # Example generation
    # ------------------------------------------------------------------

    def observation_prompts(self, observed_entities: list[str], observed_values: list[bool], query_entity: str) -> list[str]:
        """Same observation prompt phrasings as graph_dataset.py."""
        obs_list = observed_entities
        obs_strs = [
            f"{entity} is {TRUE_LABEL if value else FALSE_LABEL}"
            for entity, value in zip(observed_entities, observed_values)
        ]

        if len(obs_list) == 1:
            prompts = [
                f"Given: {obs_strs[0]}. Query: {query_entity}",
                f"Observe {obs_strs[0]}. What is {query_entity}?",
                f"{obs_strs[0]}. Determine {query_entity}.",
                f"If {obs_strs[0]}, then what is {query_entity}?",
                f"Known fact: {obs_strs[0]}. What can you infer about {query_entity}?",
                f"Observation: {obs_strs[0]}. Predict {query_entity}.",
                f"We know that {obs_strs[0]}. Is {query_entity} true or false?",
                f"Context: {obs_strs[0]}. What is the value of {query_entity}?",
                f"With {obs_strs[0]}, what do we conclude about {query_entity}?",
                f"Premise: {obs_strs[0]}. Conclusion for {query_entity}?",
            ]
        else:
            obs_and = " and ".join(obs_strs)
            obs_comma = ", ".join(obs_strs)
            prompts = [
                f"Given: {obs_and}. Query: {query_entity}",
                f"Observe {obs_and}. What is {query_entity}?",
                f"{obs_comma}. Determine {query_entity}.",
                f"If {obs_and}, then what is {query_entity}?",
                f"Known facts: {obs_and}. What can you infer about {query_entity}?",
                f"Observations: {obs_comma}. Predict {query_entity}.",
                f"We know that {obs_and}. Is {query_entity} true or false?",
                f"Context: {obs_comma}. What is the value of {query_entity}?",
                f"With {obs_and}, what do we conclude about {query_entity}?",
                f"Premises: {obs_comma}. Conclusion for {query_entity}?",
            ]
        return prompts[: self.num_observation_templates]

    def observation_specs(self, entities: list[str]) -> list[tuple[str, list[str]]]:
        """All possible query/observation specs for one world."""
        specs = []
        for query_entity in entities:
            possible_observed = [e for e in entities if e != query_entity]
            max_size = min(self.max_obs_size, len(possible_observed))
            for obs_size in range(1, max_size + 1):
                for observed_entities_tuple in combinations(possible_observed, obs_size):
                    specs.append((query_entity, list(observed_entities_tuple)))
        return specs

    def sample_observation_specs(self, specs: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
        """Sample observation specs for one world according to the configured ratio."""
        if self.sample_ratio_from_worlds >= 1.0:
            return specs
        if self.sample_ratio_from_worlds <= 0.0 or not specs:
            return []
        keep = int(round(len(specs) * self.sample_ratio_from_worlds))
        keep = min(len(specs), max(1, keep))
        return self.rng.sample(specs, keep)

    def add_observation_examples(self) -> None:
        """Make training examples from sampled worlds."""
        for graph in self.graphs:
            entities = graph["entities"]
            specs = self.observation_specs(entities)
            for _world_idx in range(self.num_worlds):
                world = self.sample_world(graph)
                for query_entity, observed_entities in self.sample_observation_specs(specs):
                    observed_values = [world[e] for e in observed_entities]
                    answer = TRUE_LABEL if world[query_entity] else FALSE_LABEL
                    inference_types = self.classify_inference_types(
                        graph, observed_entities, query_entity
                    )
                    primary = inference_types[0]
                    for prompt in self.observation_prompts(
                        observed_entities, observed_values, query_entity
                    ):
                        self.examples.append(
                            {
                                "prompt": prompt,
                                "completion": boxed(answer),
                                "answer": answer,
                                "split": "train",
                                "kind": "observation",
                                "primary_inference_type": primary,
                                "inference_types": inference_types,
                                "explanation_type": None,
                                "query_entity": query_entity,
                                "observed_entities": list(observed_entities),
                                "observed_values": [
                                    TRUE_LABEL if value else FALSE_LABEL
                                    for value in observed_values
                                ],
                                "graph_idx": graph["graph_idx"],
                            }
                        )

    def add_single_entity_explanation(self, graph: dict, entity: str, explanation_type: str, prompt: str, answer: str):
        """Add one explanation about one entity."""
        split = self.entity_to_explanation_split[entity]
        self.examples.append(
            {
                "prompt": prompt,
                "completion": boxed(answer),
                "answer": answer,
                "split": split,
                "kind": "explanation",
                "primary_inference_type": "rule_explanation",
                "inference_types": ["rule_explanation"],
                "explanation_type": explanation_type,
                "query_entity": entity,
                "observed_entities": [],
                "observed_values": [],
                "graph_idx": graph["graph_idx"],
            }
        )

    def relation_between(self, graph: dict, first: str, second: str) -> str:
        """Return the simplified relationship label for an ordered pair."""
        relation = self.relation_type(graph, first, second)
        if relation == "ancestor":
            return f"{first} is an ancestor of {second}"
        if relation == "descendant":
            return f"{first} is a descendant of {second}"
        return f"{first} and {second} are independent"

    def add_pair_explanation(self, graph: dict, first: str, second: str):
        """Add one explanation about the relation between two entities."""
        answer = self.relation_between(graph, first, second)
        split = split_name_for_mentioned_entities(
            {first, second}, self.entity_to_explanation_split
        )
        for prompt_template in self.RELATIONSHIP_QUESTIONS[: self.num_explanation_templates]:
            self.examples.append(
                {
                    "prompt": prompt_template.format(P1=first, P2=second),
                    "completion": boxed(answer),
                    "answer": answer,
                    "split": split,
                    "kind": "explanation",
                    "primary_inference_type": "rule_explanation",
                    "inference_types": ["rule_explanation"],
                    "explanation_type": "relationship",
                    "query_entity": f"{first}->{second}",
                    "observed_entities": [],
                    "observed_values": [],
                    "graph_idx": graph["graph_idx"],
                }
            )

    def add_explanation_examples(self) -> None:
        """Make train/val/test explanation examples."""
        for graph in self.graphs:
            for entity in graph["entities"]:
                parents = sorted(graph["parents"][entity])
                children = sorted(graph["children"][entity])
                rule = graph["rules"][entity]

                if rule == "ROOT":
                    rule_answer = f"{entity} is a root node"
                else:
                    rule_answer = f"{entity} = {rule}({', '.join(parents)})"
                for prompt_template in self.RULE_QUESTIONS[: self.num_explanation_templates]:
                    self.add_single_entity_explanation(
                        graph,
                        entity,
                        "rule",
                        prompt_template.format(E=entity),
                        rule_answer,
                    )

                if parents:
                    for prompt_template in self.PARENT_RETRIEVAL_QUESTIONS[: self.num_explanation_templates]:
                        self.add_single_entity_explanation(
                            graph,
                            entity,
                            "parent_retrieval",
                            prompt_template.format(E=entity),
                            " and ".join(parents),
                        )
                if children:
                    for prompt_template in self.CHILD_RETRIEVAL_QUESTIONS[: self.num_explanation_templates]:
                        self.add_single_entity_explanation(
                            graph,
                            entity,
                            "child_retrieval",
                            prompt_template.format(E=entity),
                            " and ".join(children),
                        )

            # Ordered pair relationships.  We include both A->B and B->A
            # because ancestor and descendant are direction-sensitive labels.
            for first in graph["entities"]:
                for second in graph["entities"]:
                    if first != second:
                        self.add_pair_explanation(graph, first, second)

    def generate(self) -> list[dict]:
        """Generate observations and explanations."""
        self.examples = []
        self.add_observation_examples()
        self.add_explanation_examples()
        self.apply_example_caps()
        self.add_train_explanation_copies()
        self.add_instruction_examples()
        return self.examples

    def apply_example_caps(self) -> None:
        """Optionally cap observations/explanations per split and type.

        These caps are beginner-friendly controls for dataset size.  They are
        applied after generation, so the rest of the code can stay simple.
        """
        observations = [row for row in self.examples if row["kind"] == "observation"]
        explanations = [row for row in self.examples if row["kind"] == "explanation"]

        observations = self.cap_observation_rows_by_type(
            observations,
            self.max_observations,
        )
        explanations = self.cap_rows_by_type(
            explanations,
            self.max_explanations,
            lambda row: (row["split"], row["explanation_type"]),
        )

        self.examples = observations + explanations

    def cap_observation_rows_by_type(self, rows: list[dict], cap: int | None) -> list[dict]:
        """Cap observation rows so every listed inference type stays under cap."""
        if cap is None:
            return rows

        shuffled = list(rows)
        self.rng.shuffle(shuffled)
        counts = Counter()
        kept = []
        for row in shuffled:
            keys = [
                (row["split"], inference_type)
                for inference_type in row.get("inference_types") or [row["primary_inference_type"]]
            ]
            if all(counts[key] < cap for key in keys):
                kept.append(row)
                for key in keys:
                    counts[key] += 1
        return kept

    def cap_rows_by_type(self, rows: list[dict], cap: int | None, type_key) -> list[dict]:
        """Cap rows independently for each type key."""
        if cap is None:
            return rows
        by_type = defaultdict(list)
        for row in rows:
            by_type[type_key(row)].append(row)

        kept = []
        for key in sorted(by_type):
            type_rows = by_type[key]
            if len(type_rows) > cap:
                type_rows = self.rng.sample(type_rows, cap)
            kept.extend(type_rows)
        return kept

    def add_train_explanation_copies(self) -> None:
        """Add extra copies of train explanation rows."""
        if self.num_copies_explanations <= 0:
            return
        train_explanations = [
            row
            for row in self.examples
            if row["split"] == "train" and row["kind"] == "explanation"
        ]
        self.examples.extend(
            dict(row)
            for _ in range(self.num_copies_explanations)
            for row in train_explanations
        )

    @staticmethod
    def format_instruction_output(output: str) -> str:
        """Match graph_dataset.py's light boxed-answer formatting."""
        answer_match_str = "The answer is "
        if answer_match_str in output:
            answer_start = output.rfind(answer_match_str) + len(answer_match_str)
            answer = output[answer_start:].strip(".")
            return output[:answer_start] + f"\\boxed{{{answer}}}."
        return output

    def add_instruction_examples(self) -> None:
        """Mix MathInstruct rows into the train split."""
        if self.add_instruction_datapoints <= 0:
            return
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:
            raise SystemExit(
                "Instruction mixing needs the HuggingFace datasets package. "
                "Install datasets or rerun with --add-instruction-datapoints 0."
            ) from exc

        instruct_dataset = hf_load_dataset("TIGER-Lab/MathInstruct", split="train")
        num_rows = min(self.add_instruction_datapoints, len(instruct_dataset))
        instruct_dataset = instruct_dataset.select(range(num_rows))

        for row in instruct_dataset:
            prompt = (
                row["instruction"]
                + "\nGive the answer in the format: \\boxed{answer}."
            )
            output = self.format_instruction_output(row["output"])
            self.examples.append(
                {
                    "prompt": prompt,
                    "completion": output,
                    "answer": output,
                    "split": "train",
                    "kind": "instruction",
                    "primary_inference_type": "instruct",
                    "inference_types": ["instruct"],
                    "explanation_type": None,
                    "query_entity": "N/A",
                    "observed_entities": [],
                    "observed_values": [],
                    "graph_idx": -1,
                }
            )

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def write_jsonl(self, output_dir: Path) -> None:
        """Write train/val/test JSONL files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        by_split = {"train": [], "val": [], "test": []}
        for example in self.examples:
            by_split[example["split"]].append(example)

        for split, rows in by_split.items():
            path = output_dir / f"{split}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, sort_keys=True) + "\n")

        metadata = {
            "num_graphs": self.num_graphs,
            "num_entities": self.num_entities,
            "edge_prob": self.edge_prob,
            "node_or_prob": self.node_or_prob,
            "num_worlds": self.num_worlds,
            "sample_ratio_from_worlds": self.sample_ratio_from_worlds,
            "root_prior": self.root_prior,
            "max_obs_size": self.max_obs_size,
            "max_observations": self.max_observations,
            "max_observations_unit": "per split and inference_type",
            "max_explanations": self.max_explanations,
            "max_explanations_unit": "per split and explanation_type before train copies",
            "num_copies_explanations": self.num_copies_explanations,
            "add_instruction_datapoints": self.add_instruction_datapoints,
            "heldout_scope": self.heldout_scope,
            "split_counts": {split: len(rows) for split, rows in by_split.items()},
            "kind_counts": dict(Counter(row["kind"] for row in self.examples)),
            "explanation_split_by_entity": self.entity_to_explanation_split,
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def plot_graphs(self, output_path: Path) -> None:
        """Draw all graphs using the graph_dataset.py hierarchical style."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.lines as mlines
            import matplotlib.patches as mpatches
            import networkx as nx
        except ImportError as exc:
            raise SystemExit(
                "Graph generation succeeded, but visualization needs matplotlib "
                "and networkx. Install them or rerun without --visualize."
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        entities = [entity for graph in self.graphs for entity in graph["entities"]]
        children = {
            entity: graph["children"][entity]
            for graph in self.graphs
            for entity in graph["entities"]
        }
        parents = {
            entity: graph["parents"][entity]
            for graph in self.graphs
            for entity in graph["entities"]
        }
        node_rules = {
            entity: graph["rules"][entity]
            for graph in self.graphs
            for entity in graph["entities"]
        }

        graph = nx.DiGraph()
        graph.add_nodes_from(entities)
        for parent in entities:
            for child in children[parent]:
                graph.add_edge(parent, child)

        split_colours = {
            "train": "#6baed6",
            "val": "#74c476",
            "test": "#fd8d3c",
        }
        node_colour_by_entity = {
            entity: split_colours[self.entity_to_explanation_split[entity]]
            for entity in graph.nodes()
        }

        roots = {entity for entity in entities if not parents[entity]}
        linewidth_by_entity = {
            entity: 2.5 if entity in roots else 0.8
            for entity in graph.nodes()
        }

        pos = {}
        x_cursor = 0
        node_sep = 2.0
        comp_sep = 3.0
        graph_label_positions = []

        for simple_graph in sorted(self.graphs, key=lambda item: item["graph_idx"]):
            comp_nodes = sorted(simple_graph["entities"])
            depth = {node: 0 for node in comp_nodes}
            changed = True
            while changed:
                changed = False
                for node in comp_nodes:
                    for parent in parents[node]:
                        if parent in depth and depth[parent] + 1 > depth[node]:
                            depth[node] = depth[parent] + 1
                            changed = True

            levels = defaultdict(list)
            for node in comp_nodes:
                levels[depth[node]].append(node)
            max_width = max(len(nodes) for nodes in levels.values())
            comp_half_width = (max_width - 1) / 2 * node_sep
            x_center = x_cursor + comp_half_width
            graph_label_positions.append((simple_graph["graph_idx"] + 1, x_center))

            for d, nodes in sorted(levels.items()):
                width = len(nodes)
                for i, node in enumerate(sorted(nodes)):
                    x = x_center + (i - (width - 1) / 2) * node_sep
                    pos[node] = (x, -d * node_sep)
            x_cursor = x_center + comp_half_width + comp_sep

        fig_width = max(10, len(entities) * 0.6)
        fig_height = max(6, len(entities) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        and_nodes = [entity for entity in graph.nodes() if node_rules[entity] == "AND"]
        circle_nodes = [entity for entity in graph.nodes() if entity not in and_nodes]
        for nodelist, shape in ((circle_nodes, "o"), (and_nodes, "s")):
            if not nodelist:
                continue
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodelist,
                ax=ax,
                node_color=[node_colour_by_entity[e] for e in nodelist],
                node_shape=shape,
                node_size=800,
                linewidths=[linewidth_by_entity[e] for e in nodelist],
                edgecolors="#333333",
            )
        nx.draw_networkx_labels(
            graph,
            pos,
            ax=ax,
            font_size=8,
            font_color="white",
            font_weight="bold",
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            edge_color="#555555",
            width=1.2,
            connectionstyle="arc3,rad=0.05",
        )

        if pos:
            top_y = max(y for _, y in pos.values())
            for graph_idx, x_center in graph_label_positions:
                ax.text(
                    x_center,
                    top_y + node_sep,
                    f"Graph {graph_idx}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#333333",
                )

        legend_handles = [
            mpatches.Patch(color=split_colours["train"], label="train expl"),
            mpatches.Patch(color=split_colours["val"], label="val expl (held out)"),
            mpatches.Patch(color=split_colours["test"], label="test expl (held out)"),
            mlines.Line2D([], [], color="none", marker="o", markerfacecolor="white",
                          markeredgecolor="#333333", markeredgewidth=2.5,
                          markersize=9, label="root node"),
            mlines.Line2D([], [], color="none", marker="o", markerfacecolor="white",
                          markeredgecolor="#333333", markersize=9, label="OR node"),
            mlines.Line2D([], [], color="none", marker="s", markerfacecolor="white",
                          markeredgecolor="#333333", markersize=9, label="AND node"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        ax.set_title("Simple graph dataset explanation split", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def format_counts(counts: Counter) -> str:
    """Compact stable rendering for printed summaries."""
    if not counts:
        return "none"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def observation_type_counts(rows: list[dict]) -> Counter:
    """Count every observation inference type, including mixed-type rows."""
    counts = Counter()
    for row in rows:
        if row["kind"] != "observation":
            continue
        for inference_type in row.get("inference_types") or [row["primary_inference_type"]]:
            counts[inference_type] += 1
    return counts


def explanation_type_counts(rows: list[dict]) -> Counter:
    """Count explanation subtypes."""
    return Counter(
        row["explanation_type"]
        for row in rows
        if row["kind"] == "explanation"
    )


def find_example(rows: list[dict], kind: str, type_name: str) -> dict | None:
    """Find one row of the requested kind/type."""
    for row in rows:
        if row["kind"] != kind:
            continue
        if kind == "observation" and type_name in row.get("inference_types", []):
            return row
        if kind == "explanation" and row["explanation_type"] == type_name:
            return row
    return None


def print_breakdowns_and_examples(examples: list[dict], output_dir: Path) -> None:
    """Print split/type breakdowns and one example for each available type."""
    by_split = {"train": [], "val": [], "test": []}
    for example in examples:
        by_split[example["split"]].append(example)

    all_observation_types = SimpleGraphDataset.INFERENCE_PRIORITY
    all_explanation_types = sorted(
        {
            row["explanation_type"]
            for row in examples
            if row["kind"] == "explanation"
        }
    )
    counts = Counter((row["split"], row["kind"]) for row in examples)

    print(f"Wrote data to {output_dir}")
    for split in ("train", "val", "test"):
        rows = by_split[split]
        split_total = len(rows)
        obs = counts[(split, "observation")]
        expl = counts[(split, "explanation")]
        instruct = counts[(split, "instruction")]
        print(
            f"  {split}: {split_total} rows "
            f"({obs} observations, {expl} explanations, {instruct} instructions)"
        )
        print(f"    observation inference types: {format_counts(observation_type_counts(rows))}")
        print(f"    explanation types: {format_counts(explanation_type_counts(rows))}")

    print("\nExamples by split/type")
    for split in ("train", "val", "test"):
        rows = by_split[split]
        print(f"  {split}:")
        for inference_type in all_observation_types:
            row = find_example(rows, "observation", inference_type)
            if row is None:
                print(f"    observation {inference_type}: none")
            else:
                print(
                    f"    observation {inference_type} "
                    f"(types={row['inference_types']}): {row['prompt']} -> {row['completion']}"
                )
        for explanation_type in all_explanation_types:
            row = find_example(rows, "explanation", explanation_type)
            if row is None:
                print(f"    explanation {explanation_type}: none")
            else:
                print(
                    f"    explanation {explanation_type}: "
                    f"{row['prompt']} -> {row['completion']}"
                )
        instruction = next((row for row in rows if row["kind"] == "instruction"), None)
        if instruction is not None:
            print(
                f"    instruction: {instruction['prompt']} -> {instruction['completion']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("simple_graph_data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-graphs", type=int, default=10)
    parser.add_argument("--num-entities", type=int, default=20)
    parser.add_argument("--edge-prob", type=float, default=0.2)
    parser.add_argument("--node-or-prob", type=float, default=1.0)
    parser.add_argument("--num-worlds", type=int, default=1000)
    parser.add_argument("--sample-ratio-from-worlds", type=float, default=0.01)
    parser.add_argument("--root-prior", type=float, default=0.5)
    parser.add_argument("--max-obs-size", type=int, default=2)
    parser.add_argument("--num-observation-templates", type=int, default=10)
    parser.add_argument("--num-explanation-templates", type=int, default=10)
    parser.add_argument(
        "--max-observations",
        type=int,
        default=None,
        help="Maximum observation rows per split and inference type.",
    )
    parser.add_argument(
        "--max-explanations",
        type=int,
        default=None,
        help="Maximum explanation rows per split and explanation type before train copies.",
    )
    parser.add_argument(
        "--num-copies-explanations",
        type=int,
        default=4,
        help="Extra copies of every train explanation row to append after caps.",
    )
    parser.add_argument(
        "--add-instruction-datapoints",
        type=int,
        default=2500,
        help="Number of TIGER-Lab/MathInstruct rows to add to train.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--heldout-scope", choices=["graphs", "nodes"], default="nodes")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Have alternate trials
    for trial in range(20):

        # Create an instance of a directory
        trial_output_dir = args.output_dir / ("trial_" + str(trial)) 
        trial_output_dir.mkdir(parents = True, exist_ok = True)

        dataset = SimpleGraphDataset(
            num_graphs=args.num_graphs,
            num_entities=args.num_entities,
            edge_prob=args.edge_prob,
            node_or_prob=args.node_or_prob,
            num_worlds=args.num_worlds,
            sample_ratio_from_worlds=args.sample_ratio_from_worlds,
            root_prior=args.root_prior,
            max_obs_size=args.max_obs_size,
            num_observation_templates=args.num_observation_templates,
            num_explanation_templates=args.num_explanation_templates,
            max_observations=args.max_observations,
            max_explanations=args.max_explanations,
            num_copies_explanations=args.num_copies_explanations,
            add_instruction_datapoints=args.add_instruction_datapoints,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            heldout_scope=args.heldout_scope,

            # Incrementing from a random seed for reproducibility
            seed = args.seed + trial,
        )
        
        dataset.generate()
        dataset.write_jsonl(trial_output_dir)

        if args.visualize:
            dataset.plot_graphs(trial_output_dir / "graphs.png")

        print_breakdowns_and_examples(dataset.examples, trial_output_dir)


if __name__ == "__main__":
    main()
