"""Metropolis-Hastings sampler for Decision Tables."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from numba import njit
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor import config

import pymc as pm
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars, make_shared_replacements
from pytensor.tensor.variable import Variable

from pymc_bart.bart import BARTRV
from pymc_bart.decision_table import DecisionTable, DecisionTableNode
from pymc_bart.split_rules import ContinuousSplitRule, SplitRule
from pymc_bart.utils import _encode_vi


class MHDecisionTableMove:
    """Base class for Decision Table moves."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
        rng: np.random.Generator,
        context: dict | None = None,
    ) -> tuple[DecisionTable, float, dict | None]:
        """
        Propose a new tree structure.

        Parameters
        ----------
        table : DecisionTable
            Current decision table
        X : npt.NDArray
            Input data
        Y : npt.NDArray
            Response variable
        leaf_sd : float
            Standard deviation for leaf values
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        tuple[DecisionTable, float, dict | None]
            New table, log Hastings ratio, and optional metadata
        """
        raise NotImplementedError


class GrowMove(MHDecisionTableMove):
    """Grow move: expand a leaf node into a split node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
        rng: np.random.Generator,
        context: dict | None = None,
    ) -> tuple[DecisionTable, float, dict | None]:
        """Propose growing a random leaf node."""
        new_table = table.copy()
        leaf_nodes = new_table.get_leaf_nodes(with_depth=True)

        if not leaf_nodes:
            return new_table, -np.inf, None

        # Select random leaf node
        leaf_idx = _select_leaf_index(leaf_nodes, rng, context)
        leaf_node, depth = leaf_nodes[leaf_idx]

        node_mask = _get_node_mask(
            new_table, leaf_node, X, cache=context.get("mask_cache") if context else None
        )
        if node_mask is None or not np.any(node_mask):
            return new_table, -np.inf, None

        split_var, split_value = new_table.get_level_predicate(depth)
        if split_var is None or split_value is None:
            split_var = rng.integers(0, X.shape[1])
            feature_splits = context.get("feature_splits") if context else None
            available_splits = _get_cached_split_candidates(
                X,
                split_var,
                node_mask,
                feature_splits[split_var] if feature_splits else None,
            )
            if available_splits.size == 0:
                return new_table, -np.inf, None

            split_value_raw = table.split_rules[split_var].get_split_value(available_splits)
            if split_value_raw is None:
                return new_table, -np.inf, None
            split_value = _ensure_split_array(split_value_raw)
        else:
            split_value = split_value.copy()

        split_rule = table.split_rules[split_var]
        division = _split_decision(split_rule, X[:, split_var], split_value)

        left_mask = node_mask & division
        right_mask = node_mask & (~division)

        if not left_mask.any() or not right_mask.any():
            return new_table, -np.inf, None

        left_value = _draw_leaf_value(Y, leaf_sd, left_mask, rng)
        right_value = _draw_leaf_value(Y, leaf_sd, right_mask, rng)
        old_value = leaf_node.value.copy()

        # Grow the leaf
        new_table.grow_leaf_node(
            leaf_node=leaf_node,
            selected_predictor=split_var,
            split_value=np.array([split_value]),
            left_value=left_value,
            right_value=right_value,
            left_nvalue=int(left_mask.sum()),
            right_nvalue=int(right_mask.sum()),
            depth=depth,
        )

        # Compute Hastings ratio
        n_leaf_nodes = len(leaf_nodes)
        n_split_nodes = new_table.count_split_nodes()

        log_alpha = np.log(max(n_split_nodes, 1)) - np.log(n_leaf_nodes)

        metadata = {
            "type": "grow",
            "node_mask": node_mask.copy(),
            "left_mask": left_mask.copy(),
            "right_mask": right_mask.copy(),
            "old_value": old_value,
            "left_value": left_value,
            "right_value": right_value,
        }

        return new_table, log_alpha, metadata


class PruneMove(MHDecisionTableMove):
    """Prune move: collapse a split node into a leaf node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
        rng: np.random.Generator,
        context: dict | None = None,
    ) -> tuple[DecisionTable, float, dict | None]:
        """Propose pruning a random split node."""
        new_table = table.copy()

        # Get all split nodes
        split_nodes = new_table.get_split_nodes(with_depth=True)

        if not split_nodes:
            return new_table, -np.inf, None

        n_split_nodes_before = len(split_nodes)

        # Select random split node
        split_idx = rng.integers(0, len(split_nodes))
        node_to_prune, _ = split_nodes[split_idx]

        # Check if both children are leaves
        if not all(child.is_leaf_node() for child in node_to_prune.children.values()):
            return new_table, -np.inf, None

        node_mask = _get_node_mask(
            new_table, node_to_prune, X, cache=context.get("mask_cache") if context else None
        )
        if node_mask is None or not node_mask.any():
            return new_table, -np.inf, None

        split_var = node_to_prune.idx_split_variable
        split_value = node_to_prune.value.copy()
        left_child = node_to_prune.children.get(0)
        right_child = node_to_prune.children.get(1)

        if left_child is None or right_child is None:
            return new_table, -np.inf, None

        division = _split_decision(
            table.split_rules[split_var], X[:, split_var], split_value
        )
        left_mask = node_mask & division
        right_mask = node_mask & (~division)

        if not left_mask.any() or not right_mask.any():
            return new_table, -np.inf, None

        # Draw new leaf value
        new_leaf_value = _draw_leaf_value(Y, leaf_sd, node_mask, rng)

        # Prune: convert split node to leaf
        new_table.prune_node(
            node=node_to_prune,
            new_value=new_leaf_value,
            nvalue=int(node_mask.sum()),
        )

        # Compute Hastings ratio (reverse grow selects among new leaves)
        n_leaf_nodes_after = new_table.count_leaf_nodes()
        if n_leaf_nodes_after <= 0 or n_split_nodes_before <= 0:
            return new_table, -np.inf, None

        log_alpha = np.log(n_leaf_nodes_after) - np.log(n_split_nodes_before)

        metadata = {
            "type": "prune",
            "node_mask": node_mask.copy(),
            "left_mask": left_mask.copy(),
            "right_mask": right_mask.copy(),
            "left_value": left_child.value.copy(),
            "right_value": right_child.value.copy(),
            "new_value": new_leaf_value,
        }

        return new_table, log_alpha, metadata


class ChangeMove(MHDecisionTableMove):
    """Change move: modify split rule of an existing split node."""

    def propose(
        self,
        table: DecisionTable,
        X: npt.NDArray,
        Y: npt.NDArray,
        leaf_sd: float,
        rng: np.random.Generator,
        context: dict | None = None,
    ) -> tuple[DecisionTable, float, dict | None]:
        """Propose changing a split variable or split value."""
        new_table = table.copy()

        # Get all split nodes
        split_nodes = new_table.get_split_nodes(with_depth=True)

        if not split_nodes:
            return new_table, -np.inf, None

        # Select random split node
        split_idx = rng.integers(0, len(split_nodes))
        node, depth = split_nodes[split_idx]

        node_mask = _get_node_mask(
            new_table, node, X, cache=context.get("mask_cache") if context else None
        )
        if node_mask is None or not node_mask.any():
            return new_table, -np.inf, None

        # Change split variable (with some probability keep the same)
        if rng.random() < 0.5:
            new_split_var = node.idx_split_variable
        else:
            new_split_var = rng.integers(0, X.shape[1])

        # Get available split values for new variable
        feature_splits = context.get("feature_splits") if context else None
        available_splits = _get_cached_split_candidates(
            X,
            new_split_var,
            node_mask,
            feature_splits[new_split_var] if feature_splits else None,
        )
        if available_splits.size == 0:
            return new_table, -np.inf, None

        # Select split value
        split_value_raw = table.split_rules[new_split_var].get_split_value(available_splits)
        if split_value_raw is None:
            return new_table, -np.inf, None
        split_value = _ensure_split_array(split_value_raw)

        split_rule = table.split_rules[new_split_var]
        division = _split_decision(split_rule, X[:, new_split_var], split_value)
        left_mask = node_mask & division
        right_mask = node_mask & (~division)

        if not left_mask.any() or not right_mask.any():
            return new_table, -np.inf, None

        # Update node + depth predicate
        new_table.update_level_predicate(
            depth=depth,
            split_variable=new_split_var,
            split_value=split_value,
        )

        # Hastings ratio = 1 (symmetric proposal)
        log_alpha = 0.0

        metadata = {
            "type": "change",
        }

        return new_table, log_alpha, metadata


class MHDecisionTableSampler(ArrayStepShared):
    """
    Metropolis-Hastings sampler for Decision Tables.

    Parameters
    ----------
    vars : list
        List of value variables for sampler
    num_tables : int
        Number of decision tables. Defaults to 50
    move_probs : tuple[float, float, float]
        Probabilities for (grow, prune, change) moves. Defaults to (0.33, 0.33, 0.34)
    move_adapt_rate : float
        Exponential moving-average rate for adaptive move probabilities.
        Must be in (0, 1]. Defaults to 0.1.
    move_prob_prior : float
        Positive prior weight added to each move score before normalization.
        Helps keep all moves selectable. Defaults to 0.05.
    leaf_sd : float
        Standard deviation for leaf values. Defaults to 1.0
    n_jobs : int
        Number of threads to evaluate tables in parallel (>=1). Defaults to 1.
    rng_seed : Optional[int]
        Seed used to initialize the sampler RNG. Defaults to None.
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    initial_point : Optional dict
        Initial point for sampling
    """

    name = "mh_decision_table"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "variable_inclusion": (object, []),
        "move_type": (str, []),
        "accept_rate": (float, []),
    }

    def __init__(
        self,
        vars: list[pm.Distribution] | None = None,
        num_tables: int = 50,
        move_probs: tuple[float, float, float] = (0.33, 0.33, 0.34),
        move_adapt_rate: float = 0.1,
        move_prob_prior: float = 0.05,
        leaf_sd: float = 1.0,
        n_jobs: int = 1,
        rng_seed: int | None = None,
        model: Model | None = None,
        initial_point: dict | None = None,
        **kwargs,
    ) -> None:
        model = modelcontext(model)
        
        if initial_point is None:
            initial_point = model.initial_point()
        
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)

        if vars is None:
            raise ValueError("Unable to find variables to sample")

        # Filter to only BART variables
        bart_vars = []
        for var in vars:
            rv = model.values_to_rvs.get(var)
            if rv is not None and isinstance(rv.owner.op, BARTRV):
                bart_vars.append(var)

        if not bart_vars:
            raise ValueError("No BART variables found in the provided variables")

        if len(bart_vars) > 1:
            raise ValueError(
                "MH sampler can only handle one BART variable at a time."
            )

        value_bart = bart_vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        if isinstance(self.bart.X, Variable):
            self.X = self.bart.X.eval()
        else:
            self.X = self.bart.X

        if isinstance(self.bart.Y, Variable):
            self.Y = self.bart.Y.eval()
        else:
            self.Y = self.bart.Y

        self.X = np.asarray(self.X, dtype=config.floatX)
        self.Y = np.asarray(self.Y, dtype=config.floatX)

        self.m = num_tables
        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.leaf_sd = leaf_sd
        self.feature_splits = [
            _get_available_splits(self.X, var_idx) for var_idx in range(self.num_variates)
        ]

        # Normalize move probabilities
        move_probs = np.array(move_probs)
        if np.any(move_probs <= 0):
            raise ValueError("move_probs must all be positive.")
        self.move_probs = move_probs / move_probs.sum()

        self.move_adapt_rate = float(move_adapt_rate)
        if not (0.0 < self.move_adapt_rate <= 1.0):
            raise ValueError("move_adapt_rate must be in (0, 1].")

        self.move_prob_prior = float(move_prob_prior)
        if self.move_prob_prior <= 0:
            raise ValueError("move_prob_prior must be positive.")

        # Initialize move operators
        self.moves = [GrowMove(), PruneMove(), ChangeMove()]
        self.move_names = ["grow", "prune", "change"]
        self.reverse_move_idx = [1, 0, 2]
        self.move_accept_ema = self.move_probs.astype(float).copy()
        self.rng = np.random.default_rng(rng_seed)
        self.n_jobs = max(1, int(n_jobs))

        # Initialize decision tables
        self.tables = [
            DecisionTable.new_decision_table(
                leaf_node_value=np.array([self.Y.mean() / self.m]),
                num_observations=self.num_observations,
                shape=1,
                split_rules=self.bart.split_rules
                if self.bart.split_rules
                else [ContinuousSplitRule] * self.num_variates,
            )
            for _ in range(self.m)
        ]

        self.table_predictions = [t.predict(self.X) for t in self.tables]
        self.mask_cache = [dict() for _ in range(self.m)]
        self._y_ll = self.Y.astype(np.float64, copy=False).ravel()

        self.all_tables = [[t.trim() for t in self.tables]]
        self.accept_count = 0
        self.iteration = 0
        self.model = model

        shared = make_shared_replacements(initial_point, [value_bart], model)
        self.value_bart = value_bart

        super().__init__([value_bart], shared, **kwargs)

    def astep(self, _):
        """Execute one MH step."""
        variable_inclusion = np.zeros(self.num_variates, dtype="int")
        accept_rates: list[float] = []

        seeds = self.rng.integers(
            low=0,
            high=np.iinfo(np.int64).max,
            size=self.m,
            dtype=np.int64,
        )
        tasks = [
            (idx, self.tables[idx], self.table_predictions[idx], int(seeds[idx]))
            for idx in range(self.m)
        ]

        if self.n_jobs == 1:
            results = [self._run_single_step(*task) for task in tasks]
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._run_single_step, *task) for task in tasks]
                results = [future.result() for future in futures]

        results.sort(key=lambda res: res["idx"])

        for result in results:
            idx = result["idx"]
            self.tables[idx] = result["table"]
            self.table_predictions[idx] = result["prediction"]
            self.accept_count += int(result["accepted"])
            accept_rates.append(float(result["accepted"]))
            if result["count_iteration"]:
                for var in result["split_vars"]:
                    variable_inclusion[var] += 1

        self.iteration += sum(1 for res in results if res["count_iteration"])
        self._update_move_probabilities(results)

        # Store all tables for posterior inference
        self.all_tables.append([t.trim() for t in self.tables])

        # Compute ensemble predictions
        ensemble_pred = np.mean(np.stack(self.table_predictions, axis=0), axis=0)

        accept_rate = np.mean(accept_rates) if accept_rates else 0.0
        variable_inclusion_encoded = _encode_vi(variable_inclusion.tolist())
        last_move_idx = results[-1]["move_idx"] if results else 0

        stats = {
            "variable_inclusion": variable_inclusion_encoded,
            "move_type": self.move_names[last_move_idx],
            "accept_rate": accept_rate,
        }

        return ensemble_pred, [stats]

    def _run_single_step(
        self,
        table_idx: int,
        table: DecisionTable,
        current_prediction: npt.NDArray,
        rng_seed: int,
    ) -> dict:
        """Execute a single MH proposal for one table (optionally in parallel)."""
        rng = np.random.default_rng(rng_seed)
        move_idx = rng.choice(len(self.moves), p=self.move_probs)
        move = self.moves[move_idx]
        reverse_idx = self.reverse_move_idx[move_idx]

        context = {
            "feature_splits": self.feature_splits,
            "mask_cache": self.mask_cache[table_idx],
        }

        proposed_table, log_hastings, move_metadata = move.propose(
            table,
            self.X,
            self.Y,
            self.leaf_sd,
            rng,
            context,
        )

        if log_hastings == -np.inf:
            return {
                "idx": table_idx,
                "table": table,
                "prediction": current_prediction,
                "accepted": 0,
                "move_idx": move_idx,
                "split_vars": [],
                "count_iteration": False,
            }

        new_prediction = self._apply_prediction_update(
            current_prediction, move_metadata
        )
        if new_prediction is None:
            new_prediction = proposed_table.predict(self.X)
        log_likelihood_ratio = self._compute_log_likelihood_ratio(
            current_prediction, new_prediction
        )

        log_move_ratio = np.log(self.move_probs[reverse_idx]) - np.log(
            self.move_probs[move_idx]
        )
        log_alpha = log_likelihood_ratio + log_hastings + log_move_ratio
        accepted = int(np.log(rng.random()) < log_alpha)

        final_table = proposed_table if accepted else table
        final_prediction = new_prediction if accepted else current_prediction
        if accepted:
            self.mask_cache[table_idx].clear()
        split_vars = self._get_split_variables(final_table)

        return {
            "idx": table_idx,
            "table": final_table,
            "prediction": final_prediction,
            "accepted": accepted,
            "move_idx": move_idx,
            "split_vars": split_vars,
            "count_iteration": True,
        }

    def _compute_log_likelihood_ratio(
        self,
        old_pred: npt.NDArray,
        new_pred: npt.NDArray,
    ) -> float:
        """Compute log likelihood ratio for MH acceptance."""
        old_flat = np.asarray(old_pred, dtype=np.float64).ravel()
        new_flat = np.asarray(new_pred, dtype=np.float64).ravel()

        if old_flat.shape[0] != self._y_ll.shape[0] or new_flat.shape[0] != self._y_ll.shape[0]:
            raise ValueError(
                "Predictions and observations must share the same flattened size."
            )

        return _log_likelihood_ratio_numba(
            self._y_ll,
            old_flat,
            new_flat,
            float(self.leaf_sd),
        )

    def _get_split_variables(self, table: DecisionTable) -> list[int]:
        """Get all split variables used in the table."""
        split_vars = []

        def _traverse(node: DecisionTableNode):
            if node.is_split_node():
                split_vars.append(node.idx_split_variable)
                for child in node.children.values():
                    _traverse(child)

        _traverse(table.root)
        return split_vars

    def _apply_prediction_update(
        self,
        current_prediction: npt.NDArray,
        metadata: dict | None,
    ) -> npt.NDArray | None:
        """Return updated prediction using localized move metadata."""
        if metadata is None:
            return None

        move_type = metadata.get("type")
        if move_type not in {"grow", "prune"}:
            return None

        new_pred = np.array(current_prediction, copy=True)
        flat_pred = new_pred.reshape(-1)
        node_mask = metadata["node_mask"]
        left_mask = metadata["left_mask"]
        right_mask = metadata["right_mask"]

        if move_type == "grow":
            old_value = float(np.squeeze(metadata["old_value"]))
            left_value = float(np.squeeze(metadata["left_value"]))
            right_value = float(np.squeeze(metadata["right_value"]))

            flat_pred[node_mask] -= old_value
            flat_pred[left_mask] += left_value
            flat_pred[right_mask] += right_value
            return flat_pred.reshape(new_pred.shape)

        if move_type == "prune":
            new_value = float(np.squeeze(metadata["new_value"]))
            left_value = float(np.squeeze(metadata["left_value"]))
            right_value = float(np.squeeze(metadata["right_value"]))

            flat_pred[left_mask] += new_value - left_value
            flat_pred[right_mask] += new_value - right_value
            return flat_pred.reshape(new_pred.shape)

        return None

    def _update_move_probabilities(self, results: list[dict]) -> None:
        """Adapt move probabilities using recent acceptance outcomes."""
        if not results:
            return

        adapt_rate = self.move_adapt_rate
        decay = 1.0 - adapt_rate

        for result in results:
            move_idx = result.get("move_idx")
            if move_idx is None:
                continue

            accepted = float(result.get("accepted", 0))
            current = self.move_accept_ema[move_idx]
            self.move_accept_ema[move_idx] = decay * current + adapt_rate * accepted

        scores = self.move_accept_ema + self.move_prob_prior
        total = float(scores.sum())
        if total <= 0:
            return
        self.move_probs = scores / total

    @staticmethod
    def competence(var: pm.Distribution, has_grad: bool) -> Competence:
        """MH sampler is suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    @staticmethod
    def _make_update_stats_functions():
        def update_stats(step_stats):
            return {
                key: step_stats[key]
                for key in ("variable_inclusion", "move_type", "accept_rate")
            }

        return (update_stats,)


def _select_leaf_index(
    leaf_nodes: list[tuple[DecisionTableNode, int]],
    rng: np.random.Generator,
    context: dict | None,
) -> int:
    """Select leaf index with weights favoring populous but shallower leaves."""
    if not leaf_nodes:
        raise ValueError("No leaf nodes available for selection.")

    if context is None or context.get("disable_smart_leaf"):
        return int(rng.integers(0, len(leaf_nodes)))

    weights = np.array(
        [max(node.nvalue, 1) / (1.0 + depth) for node, depth in leaf_nodes],
        dtype=float,
    )
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return int(rng.integers(0, len(leaf_nodes)))
    weights /= total
    return int(rng.choice(len(leaf_nodes), p=weights))


def _get_cached_split_candidates(
    X: npt.NDArray,
    var_idx: int,
    mask: npt.NDArray | None,
    cached_values: npt.NDArray | None,
) -> npt.NDArray:
    """Return candidate split values using cached global uniques when possible."""
    column = X[:, var_idx]
    if mask is None:
        if cached_values is not None and cached_values.size:
            return cached_values
        return _get_available_splits(X, var_idx)

    mask = _normalize_mask(mask, column.shape[0])
    values = column[mask]
    values = values[~np.isnan(values)]
    if values.size <= 1:
        return np.array([])
    if cached_values is None or cached_values.size == 0:
        return np.unique(values)

    min_value = values.min()
    max_value = values.max()
    valid = cached_values[(cached_values > min_value) & (cached_values < max_value)]
    return valid


def _get_available_splits(
    X: npt.NDArray, var_idx: int, mask: npt.NDArray | None = None
) -> npt.NDArray:
    """Get available split values for a variable."""
    values = X[:, var_idx]
    if mask is not None:
        mask = _normalize_mask(mask, values.shape[0])
        values = values[mask]
    values = values[~np.isnan(values)]
    if values.size == 0:
        return values
    return np.unique(values)


def _draw_leaf_value(
    Y: npt.NDArray,
    leaf_sd: float,
    mask: npt.NDArray | None,
    rng: np.random.Generator,
) -> npt.NDArray:
    """Draw a leaf value from normal distribution."""
    if mask is not None and mask.any():
        mask = _normalize_mask(mask, Y.shape[0])
        target = Y[mask]
    else:
        target = Y
    return np.array([np.mean(target) + rng.normal(0.0, leaf_sd)])


def _get_node_mask(
    table: DecisionTable,
    target_node: DecisionTableNode,
    X: npt.NDArray,
    cache: dict | None = None,
) -> npt.NDArray | None:
    """Return boolean mask of observations reaching the provided node."""
    node_path = None
    if cache is not None:
        node_path = _get_node_path(table, target_node)
        cached = cache.get(node_path)
        if cached is not None:
            return cached

    split_rules = table.split_rules
    n_obs = X.shape[0]

    def _traverse(node: DecisionTableNode, mask: npt.NDArray) -> npt.NDArray | None:
        mask = _normalize_mask(mask, n_obs)
        if node is target_node:
            return mask
        if node.is_leaf_node():
            return None

        split_var = node.idx_split_variable
        split_value = node.value
        division = _split_decision(split_rules[split_var], X[:, split_var], split_value)

        left_mask = mask & division
        right_mask = mask & (~division)

        if 0 in node.children:
            result = _traverse(node.children[0], left_mask)
            if result is not None:
                return result
        if 1 in node.children:
            result = _traverse(node.children[1], right_mask)
            if result is not None:
                return result
        return None

    full_mask = np.ones(n_obs, dtype=bool)
    result = _traverse(table.root, full_mask)
    if result is None:
        return None
    result = _normalize_mask(result, n_obs)
    if cache is not None and node_path is not None:
        cache[node_path] = result
    return result


def _get_node_path(table: DecisionTable, target_node: DecisionTableNode) -> tuple | None:
    """Return tuple describing path from root to target node."""
    stack: list[tuple[DecisionTableNode, tuple]] = [(table.root, ())]
    while stack:
        node, path = stack.pop()
        if node is target_node:
            return path
        for child_idx, child in node.children.items():
            stack.append((child, path + (child_idx,)))
    return None


def _ensure_split_array(value) -> npt.NDArray:
    """Ensure split values are stored as numpy arrays."""
    if isinstance(value, np.ndarray):
        return value.copy()
    arr = np.array(value, copy=True)
    if arr.ndim == 0:
        arr = arr[None]
    return arr


def _normalize_mask(mask: npt.NDArray, length: int) -> npt.NDArray:
    """Ensure mask is 1-D boolean array of requested length."""
    mask_arr = np.asarray(mask, dtype=bool)
    mask_arr = np.squeeze(mask_arr)
    mask_arr = mask_arr.reshape(-1)
    if mask_arr.size != length:
        raise ValueError(
            f"Mask has size {mask_arr.size}, expected {length}. "
            "Split rule produced incompatible shape."
        )
    return mask_arr


def _split_decision(
    split_rule: SplitRule, feature_values: npt.NDArray, split_value: npt.NDArray
) -> npt.NDArray:
    """Evaluate split rule and normalize mask shape."""
    division = split_rule.divide(feature_values, split_value)
    return _normalize_mask(division, feature_values.shape[0])


@njit(cache=True, fastmath=True)
def _log_likelihood_ratio_numba(
    y: np.ndarray,
    old_pred: np.ndarray,
    new_pred: np.ndarray,
    leaf_sd: float,
) -> float:
    """Numba-accelerated log-likelihood ratio."""
    inv_var = 1.0 / (leaf_sd * leaf_sd)
    sse_old = 0.0
    sse_new = 0.0
    for i in range(y.size):
        diff_old = y[i] - old_pred[i]
        diff_new = y[i] - new_pred[i]
        sse_old += diff_old * diff_old
        sse_new += diff_new * diff_new
    return 0.5 * (sse_old - sse_new) * inv_var
