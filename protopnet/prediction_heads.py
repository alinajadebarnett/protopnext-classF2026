import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class LinearClassPrototypePredictionHead(nn.Module):
    def __init__(
        self,
        prototype_class_identity: torch.Tensor,
        incorrect_class_connection: float = -0.5,
        k_for_topk: int = 1,
    ):
        super(LinearClassPrototypePredictionHead, self).__init__()

        self.num_classes = prototype_class_identity.shape[1]
        self.incorrect_class_connection = incorrect_class_connection
        self.k_for_topk = k_for_topk
        # FIXME - this absolutely should be persistent
        self.register_buffer(
            "prototype_class_identity", prototype_class_identity, persistent=False
        )

        self.num_prototypes = prototype_class_identity.shape[0]
        self.class_connection_layer = nn.Linear(
            self.num_prototypes,
            self.num_classes,
            bias=False,
        )

        self.__set_last_layer_incorrect_connection()

    def __set_last_layer_incorrect_connection(self):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """

        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = self.incorrect_class_connection
        self.class_connection_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def forward(
        self,
        prototype_activations: torch.Tensor,
        **kwargs,
    ):
        # TODO: Update prototype_activations to be

        _activations = prototype_activations.view(
            prototype_activations.shape[0], prototype_activations.shape[1], -1
        )

        # When k=1, this reduces to the maximum
        k_for_topk = min(self.k_for_topk, _activations.shape[-1])
        topk_activations, _ = torch.topk(_activations, k_for_topk, dim=-1)
        similarity_score_to_each_prototype = torch.mean(topk_activations, dim=-1)

        logits = self.class_connection_layer(similarity_score_to_each_prototype)

        output_dict = {"logits": logits}

        if (
            "return_similarity_score_to_each_prototype" in kwargs
            and kwargs["return_similarity_score_to_each_prototype"]
        ) or (
            "return_incorrect_class_prototype_activations" in kwargs
            and kwargs["return_incorrect_class_prototype_activations"]
        ):
            output_dict[
                "similarity_score_to_each_prototype"
            ] = similarity_score_to_each_prototype

        if (
            "return_offsets" in kwargs
            and kwargs["return_offsets"]
        ):
            output_dict["offsets"] = kwargs["offsets"]

        return output_dict


class PrototypePredictionHead(LinearClassPrototypePredictionHead):
    pass


class ProtoPoolPredictionHead(nn.Module):
    """
    Prediction head for ProtoPool (https://arxiv.org/abs/2112.02902).

    Each class has `num_descriptive` slots. Each slot picks one of the shared prototypes through
    the learned `proto_presence` tensor ([C, P, N]). Distances are mixed through the assignment,
    turned into a focal similarity (log similarity of the min distance minus that of the mean
    distance), and the C * N slot similarities feed a linear layer to C classes.

    Assignment modes:
        - hard_assignment: one-hot of `slot_prototype_index` (set at projection time)
        - gumbel_scale == 0: softmax(proto_presence)
        - training: gumbel_softmax(proto_presence * gumbel_scale, tau)
        - eval: softmax(proto_presence * gumbel_scale / tau), the noise-free counterpart
    """

    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        num_descriptive: int,
        tau: float = 0.5,
        epsilon_val: float = 1e-4,
        incorrect_class_connection: float = 0.0,
    ):
        super(ProtoPoolPredictionHead, self).__init__()
        self.num_classes = num_classes
        self.num_descriptive = num_descriptive
        self.tau = tau
        self.epsilon_val = epsilon_val
        self.incorrect_class_connection = incorrect_class_connection

        self.gumbel_scale = 0.0
        self.hard_assignment = False

        self.proto_presence = nn.Parameter(
            torch.zeros(num_classes, num_prototypes, num_descriptive)
        )
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)

        slot_class_identity = torch.zeros(num_classes * num_descriptive, num_classes)
        for j in range(num_classes * num_descriptive):
            slot_class_identity[j, j // num_descriptive] = 1
        self.register_buffer("slot_class_identity", slot_class_identity)

        self.register_buffer(
            "slot_prototype_index",
            torch.zeros(num_classes, num_descriptive, dtype=torch.long),
        )

        self.class_connection_layer = nn.Linear(
            num_classes * num_descriptive, num_classes, bias=False
        )
        positive_one_weights_locations = torch.t(slot_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        self.class_connection_layer.weight.data.copy_(
            positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    @property
    def num_prototypes(self) -> int:
        return self.proto_presence.shape[1]

    def _distance_to_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.log((distances + 1) / (distances + self.epsilon_val))

    def slot_assignment(self) -> torch.Tensor:
        """
        Returns: the effective slot assignment of shape [C, P, N], normalized over P.
        """
        if self.hard_assignment:
            return (
                F.one_hot(self.slot_prototype_index, num_classes=self.num_prototypes)
                .permute(0, 2, 1)
                .to(self.proto_presence.dtype)
            )
        if self.gumbel_scale == 0:
            return torch.softmax(self.proto_presence, dim=1)
        if self.training:
            return F.gumbel_softmax(
                self.proto_presence * self.gumbel_scale, tau=self.tau, dim=1
            )
        return torch.softmax(self.proto_presence * self.gumbel_scale / self.tau, dim=1)

    def slot_prototype_indices(self) -> torch.Tensor:
        """
        Returns: [C, N] index of the prototype used by each slot of each class.
        """
        if self.hard_assignment:
            return self.slot_prototype_index
        return torch.argmax(self.proto_presence, dim=1)

    def set_hard_assignment(self, slot_prototype_index: torch.Tensor = None):
        """
        Freezes the assignment to the argmax of proto_presence (or the given [C, N] indices).
        """
        if slot_prototype_index is None:
            slot_prototype_index = torch.argmax(self.proto_presence.detach(), dim=1)
        self.slot_prototype_index = slot_prototype_index.to(
            device=self.proto_presence.device, dtype=torch.long
        )
        self.hard_assignment = True

    def effective_class_connections(self) -> torch.Tensor:
        """
        Returns: [C, P] class connection weights per prototype under the hard (argmax) assignment.
            With hard assignment, logits == similarity_score_to_each_prototype @ W_eff.T.
        """
        weight = self.class_connection_layer.weight.view(
            -1, self.num_classes, self.num_descriptive
        )
        assignment = F.one_hot(
            self.slot_prototype_indices(), num_classes=self.num_prototypes
        ).to(weight.dtype)
        return torch.einsum("kcn,cnp->kp", weight, assignment)

    def prune_prototypes_by_index(self, prune_indices):
        prune_set = set(prune_indices)
        keep_indices = [i for i in range(self.num_prototypes) if i not in prune_set]

        old_to_new = torch.full(
            (self.num_prototypes,),
            -1,
            dtype=torch.long,
            device=self.slot_prototype_index.device,
        )
        old_to_new[keep_indices] = torch.arange(
            len(keep_indices), device=old_to_new.device
        )
        if self.hard_assignment:
            new_index = old_to_new[self.slot_prototype_index]
            assert (
                new_index >= 0
            ).all(), "Cannot prune prototypes that are still assigned to a slot."
            self.slot_prototype_index = new_index

        requires_grad = self.proto_presence.requires_grad
        self.proto_presence = nn.Parameter(
            self.proto_presence.data[:, keep_indices, :].clone(),
            requires_grad=requires_grad,
        )

    def forward(
        self,
        prototype_activations: torch.Tensor,
        prototype_distances: torch.Tensor = None,
        **kwargs,
    ):
        assert (
            prototype_distances is not None
        ), "ProtoPoolPredictionHead requires prototype_distances from the prototype layer."

        flat_distances = prototype_distances.flatten(start_dim=2)
        min_distances = flat_distances.amin(dim=-1)  # [B, P]
        avg_distances = flat_distances.mean(dim=-1)  # [B, P]

        proto_presence = self.slot_assignment()  # [C, P, N]

        min_mixed = torch.einsum("bp,cpn->bcn", min_distances, proto_presence)
        avg_mixed = torch.einsum("bp,cpn->bcn", avg_distances, proto_presence)
        focal_similarity = self._distance_to_similarity(
            min_mixed
        ) - self._distance_to_similarity(avg_mixed)

        logits = self.class_connection_layer(focal_similarity.flatten(start_dim=1))

        output_dict = {"logits": logits}

        if kwargs.get("return_min_distances", False):
            output_dict["min_distances"] = min_distances

        if kwargs.get("return_proto_presence", False):
            output_dict["proto_presence"] = proto_presence

        if kwargs.get("return_similarity_score_to_each_prototype", False) or kwargs.get(
            "return_incorrect_class_prototype_activations", False
        ):
            output_dict["similarity_score_to_each_prototype"] = (
                self._distance_to_similarity(min_distances)
                - self._distance_to_similarity(avg_distances)
            )

        return output_dict

    def get_extra_state(self):
        return {
            "gumbel_scale": self.gumbel_scale,
            "hard_assignment": self.hard_assignment,
            "tau": self.tau,
            "epsilon_val": self.epsilon_val,
            "num_classes": self.num_classes,
            "num_descriptive": self.num_descriptive,
            "incorrect_class_connection": self.incorrect_class_connection,
        }

    def set_extra_state(self, state):
        self.gumbel_scale = state["gumbel_scale"]
        self.hard_assignment = state["hard_assignment"]
        self.tau = state["tau"]
        self.epsilon_val = state["epsilon_val"]
        self.num_classes = state["num_classes"]
        self.num_descriptive = state["num_descriptive"]
        self.incorrect_class_connection = state["incorrect_class_connection"]


class ProtoTreePredictionHead(nn.Module):
    class Node(nn.Module):
        def __init__(self, index: int):
            super().__init__()
            self._index = index

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        @property
        def index(self) -> int:
            return self._index

        @property
        def size(self) -> int:
            raise NotImplementedError

        @property
        def nodes(self) -> set:
            return self.branches.union(self.leaves)

        @property
        def leaves(self) -> set:
            raise NotImplementedError

        @property
        def branches(self) -> set:
            raise NotImplementedError

        @property
        def nodes_by_index(self) -> dict:
            raise NotImplementedError

        @property
        def num_branches(self) -> int:
            return len(self.branches)

        @property
        def num_leaves(self) -> int:
            return len(self.leaves)

        @property
        def depth(self) -> int:
            raise NotImplementedError

    class Branch(Node):
        def __init__(
            self,
            index: int,
            left: "ProtoTreePredictionHead.Node",
            right: "ProtoTreePredictionHead.Node",
            log_probabilities: bool,
        ):
            super().__init__(index)
            self.left = left
            self.right = right

            # Flag that indicates whether probabilities or log probabilities are computed
            self._log_probabilities = log_probabilities

        def forward(self, xs: torch.Tensor, **kwargs):
            # Get the batch size
            batch_size = xs.size(0)

            # Keep a dict to assign attributes to nodes. Create one if not already existent
            node_attr = kwargs.setdefault("attr", dict())
            # In this dict, store the probability of arriving at this node.
            # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
            # and that it sets the path probability of arriving at its child
            # Therefore, if this attribute is not present this node is assumed to not have a parent.
            # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
            # The path probability is tracked for all x in the batch
            if not self._log_probabilities:
                pa = node_attr.setdefault(
                    (self, "pa"), torch.ones(batch_size, device=xs.device)
                )
            else:
                pa = node_attr.setdefault(
                    (self, "pa"), torch.ones(batch_size, device=xs.device)
                )

            # Obtain the probabilities of taking the right subtree
            ps = self.g(xs, **kwargs)  # shape: (bs,)

            if not self._log_probabilities:
                # Store decision node probabilities as node attribute
                node_attr[self, "ps"] = ps
                # Store path probabilities of arriving at child nodes as node attributes
                node_attr[self.left, "pa"] = (1 - ps) * pa
                node_attr[self.right, "pa"] = ps * pa
                # # Store alpha value for this batch for this decision node
                # node_attr[self, 'alpha'] = torch.sum(pa * ps) / torch.sum(pa)

                # Obtain the unweighted probability distributions from the child nodes
                l_dists, _ = self.left.forward(xs, **kwargs)  # shape: (bs, k)
                r_dists, _ = self.right.forward(xs, **kwargs)  # shape: (bs, k)
                # Weight the probability distributions by the decision node's output
                ps = ps.view(batch_size, 1)
                return (1 - ps) * l_dists + ps * r_dists, node_attr  # shape: (bs, k)
            else:
                # Store decision node probabilities as node attribute
                node_attr[self, "ps"] = ps

                # Store path probabilities of arriving at child nodes as node attributes
                # source: rewritten to pytorch from
                # https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py#L447-L471
                x = torch.abs(ps) + 1e-7  # add small epsilon for numerical stability
                oneminusp = torch.where(
                    x < np.log(2),
                    torch.log(-torch.expm1(-x)),
                    torch.log1p(-torch.exp(-x)),
                )

                node_attr[self.left, "pa"] = oneminusp + pa
                node_attr[self.right, "pa"] = ps + pa

                # Obtain the unweighted probability distributions from the child nodes
                l_dists, _ = self.left.forward(xs, **kwargs)  # shape: (bs, k)
                r_dists, _ = self.right.forward(xs, **kwargs)  # shape: (bs, k)

                # Weight the probability distributions by the decision node's output
                ps = ps.view(batch_size, 1)
                oneminusp = oneminusp.view(batch_size, 1)
                logs_stacked = torch.stack((oneminusp + l_dists, ps + r_dists))
                return torch.logsumexp(logs_stacked, dim=0), node_attr  # shape: (bs,)

        def g(self, xs: torch.Tensor, **kwargs):
            out_map = kwargs[
                "out_map"
            ]  # Obtain the mapping from decision nodes to conv net outputs
            conv_net_output = kwargs["conv_net_output"]  # Obtain the conv net outputs
            out = conv_net_output[
                out_map[self]
            ]  # Obtain the output corresponding to this decision node
            return out.squeeze(dim=1)

        @property
        def size(self) -> int:
            return 1 + self.left.size + self.right.size

        @property
        def leaves(self) -> set:
            return self.left.leaves.union(self.right.leaves)

        @property
        def branches(self) -> set:
            return {self}.union(self.left.branches).union(self.right.branches)

        @property
        def nodes_by_index(self) -> dict:
            return {
                self.index: self,
                **self.left.nodes_by_index,
                **self.right.nodes_by_index,
            }

        @property
        def num_branches(self) -> int:
            return 1 + self.left.num_branches + self.right.num_branches

        @property
        def num_leaves(self) -> int:
            return self.left.num_leaves + self.right.num_leaves

        @property
        def depth(self) -> int:
            return self.left.depth + 1

    class Leaf(Node):
        def __init__(
            self,
            index: int,
            num_classes: int,
            log_probabilities: bool,
            disable_derivative_free_leaf_optim,
        ):
            super().__init__(index)

            # Flag that indicates whether probabilities or log probabilities are computed
            self._log_probabilities = log_probabilities

            # Initialize the distribution parameters
            if disable_derivative_free_leaf_optim:
                # initialize with random and with gradient
                self._dist_params = nn.Parameter(
                    torch.randn(num_classes), requires_grad=True
                )
            else:
                # initialize with zeros and without gradient
                self._dist_params = nn.Parameter(
                    torch.zeros(num_classes), requires_grad=False
                )

        def forward(self, activations: torch.Tensor, **kwargs):
            # Get the batch size
            batch_size = activations.shape[0]

            # Keep a dict to assign attributes to nodes. Create one if not already existent
            node_attr = kwargs.setdefault("attr", dict())
            # In this dict, store the probability of arriving at this node.
            # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
            # and that it sets the path probability of arriving at its child
            # Therefore, if this attribute is not present this node is assumed to not have a parent.
            # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
            # The path probability is tracked for all x in the batch
            if not self._log_probabilities:
                node_attr.setdefault(
                    (self, "pa"), torch.ones(batch_size, device=activations.device)
                )
            else:
                node_attr.setdefault(
                    (self, "pa"), torch.zeros(batch_size, device=activations.device)
                )

            # Obtain the leaf distribution
            dist = self.distribution()  # shape: (k,)
            # Reshape the distribution to a matrix with one single row
            dist = dist.view(1, -1)  # shape: (1, k)
            # Duplicate the row for all x in xs
            dists = torch.cat((dist,) * batch_size, dim=0)  # shape: (bs, k)

            # Store leaf distributions as node property
            node_attr[self, "ds"] = dists

            # Return both the result of the forward pass as well as the node properties
            return dists, node_attr

        def distribution(self) -> torch.Tensor:
            eps = 1e-10
            if self._log_probabilities:
                return F.log_softmax(self._dist_params + eps, dim=0)
            else:
                # Return numerically stable softmax (see http://www.deeplearningbook.org/contents/numerical.html)
                return F.softmax(
                    self._dist_params - torch.max(self._dist_params), dim=0
                )

        @property
        def requires_grad(self) -> bool:
            return self._dist_params.requires_grad

        @requires_grad.setter
        def requires_grad(self, val: bool):
            self._dist_params.requires_grad = val

        @property
        def size(self) -> int:
            return 1

        @property
        def leaves(self) -> set:
            return {self}

        @property
        def branches(self) -> set:
            return set()

        @property
        def nodes_by_index(self) -> dict:
            return {self.index: self}

        @property
        def num_branches(self) -> int:
            return 0

        @property
        def num_leaves(self) -> int:
            return 1

        @property
        def depth(self) -> int:
            return 0

    class ProtoTree(nn.Module):
        def __init__(
            self,
            num_classes: int,
            depth: int,
            log_probabilities: bool,
            disable_derivative_free_leaf_optim,
            k_for_topk: int = 1,
        ):
            super().__init__()

            self._num_classes = num_classes

            self._root = self._init_tree(
                num_classes,
                depth,
                log_probabilities,
                disable_derivative_free_leaf_optim,
            )

            self.num_prototypes = self.num_branches

            self._parents = dict()
            self._set_parents()

            # Flag that indicates whether probabilities or log probabilities are computed
            self._log_probabilities = log_probabilities

            self.k_for_topk = k_for_topk

            # Map each decision node to an output of the feature net
            self._out_map = {
                n: i for i, n in zip(range(2 ** (depth) - 1), self.branches)
            }

        @property
        def root(self) -> "ProtoTreePredictionHead.Node":
            return self._root

        @property
        def leaves_require_grad(self) -> bool:
            return any([leaf.requires_grad for leaf in self.leaves])

        @leaves_require_grad.setter
        def leaves_require_grad(self, val: bool):
            for leaf in self.leaves:
                leaf.requires_grad = val

        def forward(
            self,
            prototype_activations: torch.Tensor,
            **kwargs,
        ) -> tuple:
            """
            PERFORM A FORWARD PASS THROUGH THE TREE GIVEN THE COMPUTED SIMILARITIES
            """

            _activations = prototype_activations.view(
                prototype_activations.shape[0], prototype_activations.shape[1], -1
            )

            # When k=1, this reduces to the maximum
            k_for_topk = min(self.k_for_topk, _activations.shape[-1])
            topk_activations, _ = torch.topk(_activations, k_for_topk, dim=-1)
            similarity_score_to_each_prototype = torch.mean(topk_activations, dim=-1)

            similarities = similarity_score_to_each_prototype

            if self._log_probabilities:
                similarities = torch.log(similarities)

            kwargs["conv_net_output"] = similarities.chunk(similarities.size(1), dim=1)

            kwargs["out_map"] = dict(self._out_map)

            # Perform a forward pass through the tree
            logits, attr = self._root.forward(_activations, **kwargs)

            # logits are (batch size, k)
            output_dict = {"logits": logits}

            if (
                "return_similarity_score_to_each_prototype" in kwargs
                and kwargs["return_similarity_score_to_each_prototype"]
            ) or (
                "return_incorrect_class_prototype_activations" in kwargs
                and kwargs["return_incorrect_class_prototype_activations"]
            ):
                output_dict[
                    "similarity_score_to_each_prototype"
                ] = similarity_score_to_each_prototype

            if (
                "return_offsets" in kwargs
                and kwargs["return_offsets"]
            ):
                output_dict["offsets"] = kwargs["offsets"]

            # Store the probability of arriving at all nodes in the decision tree
            output_dict["pa_tensor"] = {
                n.index: attr[n, "pa"].unsqueeze(1) for n in self.nodes
            }
            # Store the output probabilities of all decision nodes in the tree
            output_dict["ps"] = {
                n.index: attr[n, "ps"].unsqueeze(1) for n in self.branches
            }

            if not self._log_probabilities:
                output_dict["logits"] = torch.log(output_dict["logits"])

            return output_dict

        @property
        def depth(self) -> int:
            def d(node: ProtoTreePredictionHead.Node):
                return (
                    1
                    if isinstance(node, self.Leaf)
                    else 1 + max(d(node.left), d(node.right))
                )

            return d(self._root)

        @property
        def size(self) -> int:
            return self._root.size

        @property
        def nodes(self) -> set:
            return self._root.nodes

        @property
        def nodes_by_index(self) -> dict:
            return self._root.nodes_by_index

        @property
        def node_depths(self) -> dict:
            def _assign_depths(node, d):
                if isinstance(node, self.Leaf):
                    return {node: d}
                if isinstance(node, self.Branch):
                    return {
                        node: d,
                        **_assign_depths(node.right, d + 1),
                        **_assign_depths(node.left, d + 1),
                    }

            return _assign_depths(self._root, 0)

        @property
        def branches(self) -> set:
            return self._root.branches

        @property
        def leaves(self) -> set:
            return self._root.leaves

        @property
        def num_branches(self) -> int:
            return self._root.num_branches

        @property
        def num_leaves(self) -> int:
            return self._root.num_leaves

        def _init_tree(
            self,
            num_classes: int,
            depth: int,
            log_probabilities,
            disable_derivative_free_leaf_optim,
        ) -> "ProtoTreePredictionHead.Node":
            def _init_tree_recursive(
                i: int, d: int
            ) -> ProtoTreePredictionHead.Node:  # Recursively build the tree
                if d == depth:
                    return ProtoTreePredictionHead.Leaf(
                        i,
                        num_classes,
                        log_probabilities,
                        disable_derivative_free_leaf_optim,
                    )
                else:
                    left = _init_tree_recursive(i + 1, d + 1)
                    return ProtoTreePredictionHead.Branch(
                        i,
                        left,
                        _init_tree_recursive(i + left.size + 1, d + 1),
                        log_probabilities,
                    )

            return _init_tree_recursive(0, 0)

        def _set_parents(self) -> None:
            self._parents.clear()
            self._parents[self._root] = None

            def _set_parents_recursively(node: ProtoTreePredictionHead.Node):
                if isinstance(node, ProtoTreePredictionHead.Branch):
                    self._parents[node.right] = node
                    self._parents[node.left] = node
                    _set_parents_recursively(node.right)
                    _set_parents_recursively(node.left)
                    return
                if isinstance(node, ProtoTreePredictionHead.Leaf):
                    return  # Nothing to do here!
                raise Exception("Unrecognized node type!")

            # Set all parents by traversing the tree starting from the root
            _set_parents_recursively(self._root)

        def path_to(self, node: "ProtoTreePredictionHead.Node"):
            assert node in self.leaves or node in self.branches
            path = [node]
            while isinstance(self._parents[node], ProtoTreePredictionHead.Node):
                node = self._parents[node]
                path = [node] + path
            return path

    def __init__(
        self,
        num_classes: int,
        depth: int,
        log_probabilities: bool,
        disable_derivative_free_leaf_optim: bool,
        k_for_topk: int = 1,
        pruning_threshold: float = 0.01,
    ):
        super().__init__()
        self.prototree = ProtoTreePredictionHead.ProtoTree(
            num_classes,
            depth,
            log_probabilities,
            disable_derivative_free_leaf_optim,
            k_for_topk,
        )
        self._pruning_threshold = pruning_threshold

    def forward(self, *args, **kwargs):
        return self.prototree(*args, **kwargs)

    def batch_derivative_free_tree_update(
        self, output, target, num_batches, old_dist_params
    ):
        """
        Updating ProtoTree torch module parameters using a derivative free
        update method.

        Args:
            output: dict -
        """

        if "logits" not in output:
            # FIXME - this is a strong assumption
            log.debug(
                "No logits in output, skipping derivative free tree for non-classification epoch"
            )
            return

        with torch.no_grad():
            predicted_dist = output["logits"]
            target_dist = torch.nn.functional.one_hot(
                target, num_classes=self.prototree._num_classes
            )
            if self.prototree._log_probabilities:
                # ensure all terms of update method are in log probabilities
                target_dist = torch.log(target_dist)
            else:
                # since output of our model is always a log probability
                # (see tree forward and loss function in trainer),
                # we calculate convert the prediction back to normalized
                # proability
                predicted_dist = torch.exp(predicted_dist)

            for leaf in self.prototree.leaves:
                if self.prototree._log_probabilities:
                    # log version
                    update = torch.exp(
                        torch.logsumexp(
                            output["pa_tensor"][leaf.index]
                            + leaf.distribution()
                            + target_dist
                            - predicted_dist,
                            dim=0,
                        )
                    )
                else:
                    update = torch.sum(
                        (
                            output["pa_tensor"][leaf.index]
                            * leaf.distribution()
                            * target_dist
                        )
                        / predicted_dist,
                        dim=0,
                    )
                leaf._dist_params -= old_dist_params[leaf] / num_batches
                F.relu_(
                    leaf._dist_params
                )  # dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                leaf._dist_params += update

    def check_prune_threshold(self) -> list:
        """
        Internal method for identifying which tree nodes are market for pruning.
        Mark any node whose set of descendant leaves contains at least one leaf
        whose maximal prediction is less than the pruning threshold. For datasets
        with more classes, you may consider different pruning thresholds. To see
        usuage, look at `ProtoTreePredictionHead.prune_tree`.

        Returns:
            list: Tree nodes marked for pruning
        """

        # Collects the nodes
        def nodes_to_prune_based_on_leaf_dists_threshold(
            tree: ProtoTreePredictionHead.ProtoTree,
        ) -> list:
            to_prune_incl_possible_children = []
            for node in tree.nodes:
                if has_max_prob_lower_threshold(node):
                    # prune everything below incl this node
                    to_prune_incl_possible_children.append(node.index)
            return to_prune_incl_possible_children

        # Returns True when all the node's children have a max leaf value < threshold
        def has_max_prob_lower_threshold(node: ProtoTreePredictionHead.Node):
            if isinstance(node, ProtoTreePredictionHead.Branch):
                for leaf in node.leaves:
                    if leaf._log_probabilities:
                        if (
                            torch.max(torch.exp(leaf.distribution())).item()
                            > self._pruning_threshold
                        ):
                            return False
                    else:
                        if (
                            torch.max(leaf.distribution()).item()
                            > self._pruning_threshold
                        ):
                            return False
            elif isinstance(node, ProtoTreePredictionHead.Leaf):
                if node._log_probabilities:
                    if (
                        torch.max(torch.exp(node.distribution())).item()
                        > self._pruning_threshold
                    ):
                        return False
                else:
                    if torch.max(node.distribution()).item() > self._pruning_threshold:
                        return False
            else:
                raise Exception(
                    "This node type should not be possible. A tree has branches and leaves."
                )
            return True

        return nodes_to_prune_based_on_leaf_dists_threshold(self.prototree)

    def prune_tree(self):
        """
        Restructure tree by pruning prototypes given index of nodes
        """

        prune_node_indices = self.check_prune_threshold()

        to_prune = deepcopy(prune_node_indices)
        # remove children from prune_list of nodes that would already be pruned
        for node_idx in prune_node_indices:
            if isinstance(
                self.prototree.nodes_by_index[node_idx], ProtoTreePredictionHead.Branch
            ):
                if (
                    node_idx > 0
                ):  # parent cannot be root since root would then be removed
                    for child in self.prototree.nodes_by_index[node_idx].nodes:
                        if child.index in to_prune and child.index != node_idx:
                            to_prune.remove(child.index)

        for node_idx in to_prune:
            node = self.prototree.nodes_by_index[node_idx]
            parent = self.prototree._parents[node]
            if (
                parent.index > 0
            ):  # parent cannot be root since root would then be removed
                if node == parent.left:
                    if parent == self.prototree._parents[parent].left:
                        # make right child of parent the left child of parent of parent
                        self.prototree._parents[parent.right] = self.prototree._parents[
                            parent
                        ]
                        self.prototree._parents[parent].left = parent.right
                    elif parent == self.prototree._parents[parent].right:
                        # make right child of parent the right child of parent of parent
                        self.prototree._parents[parent.right] = self.prototree._parents[
                            parent
                        ]
                        self.prototree._parents[parent].right = parent.right
                    else:
                        raise Exception(
                            "Pruning went wrong, this should not be possible"
                        )

                elif node == parent.right:
                    if parent == self.prototree._parents[parent].left:
                        # make left child or parent the left child of parent of parent
                        self.prototree._parents[parent.left] = self.prototree._parents[
                            parent
                        ]
                        self.prototree._parents[parent].left = parent.left
                    elif parent == self.prototree._parents[parent].right:
                        # make left child of parent the right child of parent of parent
                        self.prototree._parents[parent.left] = self.prototree._parents[
                            parent
                        ]
                        self.prototree._parents[parent].right = parent.left
                    else:
                        raise Exception(
                            "Pruning went wrong, this should not be possible"
                        )
                else:
                    raise Exception("Pruning went wrong, this should not be possible")

        # Calculating the indices of prototypes being prune requires first
        # knowing which branches remain after pruning. Then, we have to
        # update the mapping of branches to prototypes. Lastly, we reset
        # indices in the mapping and pass the prototype layer the indices
        # that need to be removed.
        prune_prototype_indices = []

        # reveal the remaining branches after tree mutation
        remaining_branches = self.prototree.branches

        # expose the mapping of branches to prototypes to help inform us the
        # mapping of branches to prototypes (and thus identify what prototypes
        # will need to be prune/updated)

        # remove pruned branches while recording the prototypes associated
        # with them.
        for branch, prototype_idx in dict(self.prototree._out_map).items():
            if branch not in remaining_branches:
                # using dict copy to avoid mutating dict during iterations
                del self.prototree._out_map[branch]
                prune_prototype_indices.append(prototype_idx)

        # get remaining prototype indices / prepare to reindex them while
        # preserving their branch relationship after prototype layer prune
        sorted_branches_by_prototype_index = sorted(
            self.prototree._out_map.items(), key=lambda x: x[1]
        )

        # re-index prototypes (0,1,2,3,...), this should line up with prototypes
        for new_idx, (branch, _) in enumerate(sorted_branches_by_prototype_index):
            self.prototree._out_map[branch] = new_idx

        return sorted(prune_prototype_indices)
