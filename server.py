# -*- coding: utf-8 -*-
"""
Server-side implementation for PADAE federated learning experiments.

This version uses quarantine-based PADAE defense logic:
1. Candidate clients are trained and inspected in each round.
2. MQV and MPDD both can mark a client as abnormal.
3. First abnormal round:
   - status = warning
   - excluded from current aggregation
   - still trained and inspected in the next round
4. Consecutive abnormal rounds reaching the threshold:
   - status = removed
   - permanently excluded from training, inspection, and aggregation
5. Aggregation uses secure clients only.
6. Final testing evaluates all clients for complete reporting.

MPDD setting:
- MPDD is weight-based, not update-based.
- It compares raw local model parameter distributions.
- Only the first trainable layer is used for parameter distribution comparison.
"""


import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp

from client import train, test
from data_process import validationSet
from model import DNN

np.set_printoptions(threshold=np.inf)


class FedAvg:
    def __init__(self, args):
        """
        Initialize server, global model, and local client models.
        """
        self.args = args
        self.clients = args.clients

        self.nn = DNN(args=args, file_name="server")
        self.nns = []

        # Clients that are still allowed to be trained and inspected.
        # Warning clients remain here.
        self.candidate_clients = [i for i in range(self.args.K)]

        # Clients that passed the current round inspection and can be aggregated.
        self.secure_server = [i for i in range(self.args.K)]

        # Clients that are permanently removed.
        self.removed_clients = []

        # Number of consecutive abnormal rounds required before permanent removal.
        self.abnormal_round_threshold = getattr(
            self.args,
            "abnormal_round_threshold",
            2
        )

        self.client_status = {}

        for i in range(self.args.K):
            self.client_status[i] = {
                "client_name": self.clients[i],
                "status": "active",
                "removed_round": None,
                "removed_by": None,
                "last_ks_value": None,
                "last_val_accuracy": None,
                "last_pva_mean": None,
                "abnormal_count": 0,
                "last_abnormal_reason": None,
                "final_accuracy": None,
                "used_for_aggregation": True,
            }       
        for i in range(self.args.K):
            local_model = DNN(args=args, file_name="server")
            local_model.file_name = self.clients[i]
            self.nns.append(local_model)

    def server(self):
        """
        Run federated learning training rounds.

        Candidate clients are always trained and inspected unless permanently removed.
        Secure clients are the only clients used for aggregation.
        """
        for round_id in range(self.args.r):
            current_round = round_id + 1
            print(f"\nRound {current_round}:")

            if len(self.candidate_clients) == 0:
                raise RuntimeError(
                    "No candidate clients remain. "
                    "Please check thresholds or attack settings."
                )

            index = self.candidate_clients.copy()

            print("Candidate client indices:", index)
            print("Current secure client indices before inspection:", self.secure_server)
            print(
                "Consecutive abnormal removal threshold:",
                self.abnormal_round_threshold
            )

            # Dispatch the latest global model to all candidate clients.
            self.dispatch(index)

            # Train all candidate clients, including warning clients.
            self.client_update(index)

            # Record MQV / MPDD abnormal reasons in the current round.
            round_abnormal_reasons = {
                client_id: []
                for client_id in index
            }

            self.validation_set(
                batch=self.args.B,
                index=index,
                current_round=current_round,
                round_abnormal_reasons=round_abnormal_reasons
            )

            self.distribution_difference(
                current_round=current_round,
                index=index,
                round_abnormal_reasons=round_abnormal_reasons
            )

            self.apply_quarantine_filter(
                index=index,
                current_round=current_round,
                round_abnormal_reasons=round_abnormal_reasons
            )

            if len(self.secure_server) == 0:
                raise RuntimeError(
                    "No secure clients are available for aggregation in this round. "
                    "Please relax thresholds or check attack settings."
                )

            self.aggregation()

        return self.nn

    def dispatch(self, index):
        """
        Dispatch the latest global model weights to selected clients.
        """
        global_weights = self.nn.get_weights()

        for client_id in index:
            self.nns[client_id].set_weights(global_weights)

    def client_update(self, index):
        """
        Perform local training on candidate clients.
        """
        for client_id in index:
            self.nns[client_id] = train(
                self.args,
                self.nns[client_id],
                self.nns[client_id].file_name,
                client_id
            )

    def validation_set(
        self,
        batch,
        index,
        current_round,
        round_abnormal_reasons
    ):
        """
        Model Quality Validation (MQV).

        If KS_value is lower than ks_threshold, the client is marked abnormal
        for this round. It is not immediately removed here.
        """
        X_val, y_val = validationSet(self.args, batch)
        y_true = y_val.to_numpy(dtype=np.float32)

        ks_threshold = getattr(self.args, "ks_threshold", 0.5)

        print("KS_value:")

        for client_id in index:
            pred = self.nns[client_id].predict(
                X_val,
                batch_size=batch,
                verbose=0
            )
            # ============================================================
            # MQV: KS_value
            # ============================================================
            fpr, tpr, _ = roc_curve(y_true.ravel(), pred.ravel())
            ks_value = float(np.max(np.abs(tpr - fpr)))

            self.client_status[client_id]["last_ks_value"] = ks_value

            # ============================================================
            # Server-side validation accuracy
            # ============================================================
            pred_label = np.argmax(pred, axis=1)
            true_label = np.argmax(y_true, axis=1)

            val_accuracy = float(np.mean(pred_label == true_label))

            self.client_status[client_id]["last_val_accuracy"] = val_accuracy

            if ks_value < ks_threshold:
                round_abnormal_reasons[client_id].append("MQV")

            mqv_flag = (
                "ABNORMAL"
                if "MQV" in round_abnormal_reasons[client_id]
                else "normal"
            )

            print(
                f"  {self.clients[client_id]} "
                f"(index={client_id}) "
                f"KS_value = {ks_value:.6f}, "
                f"Val_ACC = {val_accuracy:.6f} "
                f"[{mqv_flag}]"
            )
   
    def flatten_all_layer_kernel_weights(self, model):
        """
        Flatten only the kernel weights of all trainable layers.

        This includes:
        - kernel weights of every trainable layer

        This does not include:
        - bias
        - delta_W = local_W - global_W
        """
        flattened_weights = []

        for layer in model.layers:
            layer_weights = layer.get_weights()

            if len(layer_weights) == 0:
                continue

            # layer_weights[0] is usually kernel
            kernel = layer_weights[0]

            flattened_weights.append(
                np.asarray(kernel, dtype=np.float64).flatten()
            )

        if len(flattened_weights) == 0:
            raise ValueError(
                "No kernel weights were collected from the model."
            )

        return np.concatenate(flattened_weights)
    
    def flatten_first_middle_last_layer_kernel_weights(self, model):
        """
        Flatten only the kernel weights of the first, middle, and last trainable layers.

        This is weight-based MPDD:
            W_first_layer_kernel + W_middle_layer_kernel + W_last_layer_kernel

        This does not include:
        - bias terms
        - delta_W = local_W - global_W
        """
        weighted_layers = []

        for layer in model.layers:
            layer_weights = layer.get_weights()

            if len(layer_weights) > 0:
                weighted_layers.append(layer)

        if len(weighted_layers) == 0:
            raise ValueError(
                "The model does not contain any trainable weighted layers."
            )

        if len(weighted_layers) == 1:
            selected_layers = [weighted_layers[0]]

        elif len(weighted_layers) == 2:
            selected_layers = [
                weighted_layers[0],
                weighted_layers[-1],
            ]

        else:
            middle_index = (len(weighted_layers) - 1) // 2

            selected_layers = [
                weighted_layers[0],
                weighted_layers[middle_index],
                weighted_layers[-1],
            ]

        flattened_weights = []

        for layer in selected_layers:
            layer_weights = layer.get_weights()

            if len(layer_weights) == 0:
                continue

            # layer_weights[0] is usually kernel
            kernel = layer_weights[0]

            flattened_weights.append(
                np.asarray(kernel, dtype=np.float64).flatten()
            )

        if len(flattened_weights) == 0:
            raise ValueError(
                "No kernel weights were collected from the selected layers."
            )

        return np.concatenate(flattened_weights)

    def distribution_difference(
        self,
        current_round,
        index,
        round_abnormal_reasons
    ):
        """
        Weight-based Model Parameter Distribution Detection (MPDD).

        Dataset-specific MPDD parameter extraction:
        - UNSW-NB15:
            use all-layer kernel weights
            flatten_all_layer_kernel_weights()

        - CIC-IDS2017:
            use first-middle-last-layer kernel weights
            flatten_first_middle_last_layer_kernel_weights()

        This MPDD implementation:
        - compares raw local model parameter distributions
        - uses kernel weights only
        - does not include bias terms
        - does not compute delta_W = local_W - global_W

        PVA_mean is computed among candidate clients in the current round.
        If PVA_mean is lower than or equal to pvalue_threshold, the client is
        marked abnormal for this round.
        """
        if len(index) <= 1:
            print("MPDD skipped because fewer than two candidate clients remain.")
            return

        dataset_name = getattr(self.args, "dataset", "")

        if dataset_name == "UNSW-NB15":
            weight_extractor = self.flatten_all_layer_kernel_weights
            mpdd_setting_name = "all_layer_kernel"

        elif dataset_name == "CIC-IDS2017":
            weight_extractor = self.flatten_first_middle_last_layer_kernel_weights
            mpdd_setting_name = "first_middle_last_layer_kernel"

        else:
            # Fallback setting for unsupported or newly added datasets.
            weight_extractor = self.flatten_all_layer_kernel_weights
            mpdd_setting_name = "all_layer_kernel"

        print(
            "MPDD parameter extraction setting: "
            f"dataset={dataset_name}, "
            f"method={mpdd_setting_name}"
        )

        pvalue_matrix = []

        for client_i in index:
            pvalues = []

            weight_i = weight_extractor(
                self.nns[client_i]
            )

            for client_j in index:
                weight_j = weight_extractor(
                    self.nns[client_j]
                )

                pvalue = ks_2samp(weight_i, weight_j).pvalue
                pvalues.append(pvalue)

            pvalue_matrix.append(pvalues)

        avg_pvalues = []

        for row_id, pvalues in enumerate(pvalue_matrix):
            pvalues_without_self = [
                value
                for col_id, value in enumerate(pvalues)
                if col_id != row_id
            ]

            avg_pvalue = float(np.mean(pvalues_without_self))
            avg_pvalues.append(avg_pvalue)

        pvalue_threshold = getattr(self.args, "pvalue_threshold", 0.05)

        print(f"{mpdd_setting_name}-based PVA_mean:")

        for client_id, pva_mean in zip(index, avg_pvalues):
            self.client_status[client_id]["last_pva_mean"] = pva_mean

            if pva_mean <= pvalue_threshold:
                round_abnormal_reasons[client_id].append("MPDD")

            mpdd_flag = (
                "ABNORMAL"
                if "MPDD" in round_abnormal_reasons[client_id]
                else "normal"
            )

            print(
                f"  {self.clients[client_id]} "
                f"(index={client_id}) "
                f"{mpdd_setting_name}_PVA_mean = {pva_mean:.6f} "
                f"[{mpdd_flag}]"
            )


    def apply_quarantine_filter(
        self,
        index,
        current_round,
        round_abnormal_reasons
    ):
        """
        Apply quarantine-based filtering.

        Rules:
        - Normal in this round:
            abnormal_count = 0
            status = active
            included in candidate_clients
            included in secure_server

        - Abnormal for the first time:
            abnormal_count += 1
            status = warning
            included in candidate_clients
            excluded from secure_server for this round

        - Consecutive abnormal count reaches threshold:
            status = removed
            excluded from candidate_clients
            excluded from secure_server permanently
        """
        next_candidate_clients = []
        next_secure_clients = []
        warning_clients = []
        removed_this_round = []

        for client_id in index:
            reasons = round_abnormal_reasons.get(client_id, [])

            if len(reasons) == 0:
                # Recovered or consistently normal.
                self.client_status[client_id]["abnormal_count"] = 0
                self.client_status[client_id]["last_abnormal_reason"] = None
                self.client_status[client_id]["status"] = "active"
                self.client_status[client_id]["removed_round"] = None
                self.client_status[client_id]["removed_by"] = None
                self.client_status[client_id]["used_for_aggregation"] = True

                next_candidate_clients.append(client_id)
                next_secure_clients.append(client_id)

            else:
                # Abnormal in this round.
                reason_text = "+".join(reasons)

                self.client_status[client_id]["abnormal_count"] += 1
                self.client_status[client_id]["last_abnormal_reason"] = reason_text
                self.client_status[client_id]["used_for_aggregation"] = False

                abnormal_count = self.client_status[client_id]["abnormal_count"]

                if abnormal_count >= self.abnormal_round_threshold:
                    # Permanently remove.
                    self.client_status[client_id]["status"] = "removed"
                    self.client_status[client_id]["removed_round"] = current_round
                    self.client_status[client_id]["removed_by"] = reason_text

                    if client_id not in self.removed_clients:
                        self.removed_clients.append(client_id)

                    removed_this_round.append(client_id)

                else:
                    # Quarantine: do not aggregate this round,
                    # but keep it for next-round training and inspection.
                    self.client_status[client_id]["status"] = "warning"
                    self.client_status[client_id]["removed_round"] = None
                    self.client_status[client_id]["removed_by"] = None

                    next_candidate_clients.append(client_id)
                    warning_clients.append(client_id)

        # Permanently removed clients are excluded from candidates.
        self.candidate_clients = next_candidate_clients

        # Only normal clients are aggregated in this round.
        self.secure_server = next_secure_clients

        print("Clients retained for next-round inspection:", self.candidate_clients)
        print("Clients allowed for current aggregation:", self.secure_server)

        if warning_clients:
            print("Warning clients quarantined from current aggregation:", warning_clients)

            for client_id in warning_clients:
                print(
                    f"  {self.clients[client_id]} "
                    f"(index={client_id}) abnormal_count = "
                    f"{self.client_status[client_id]['abnormal_count']} / "
                    f"{self.abnormal_round_threshold}, "
                    f"reason = {self.client_status[client_id]['last_abnormal_reason']}"
                )

        if removed_this_round:
            print("Clients permanently removed:", removed_this_round)

            for client_id in removed_this_round:
                print(
                    f"  {self.clients[client_id]} "
                    f"(index={client_id}) removed at round {current_round}, "
                    f"reason = {self.client_status[client_id]['removed_by']}"
                )

    def compute_cma_group_weights(self):
        """
        Compute CMA aggregation weights using beta and lambda.

        beta:
            proportion of high-contribution clients.

        lambda:
            total aggregation weight assigned to the high-contribution group.

        Contribution score:
            score_i = KS_value_i * PVA_mean_i

        If PVA_mean is unavailable, only KS_value is used.
        """
        if len(self.secure_server) == 0:
            raise RuntimeError("No secure clients available for CMA.")

        beta = getattr(self.args, "cma_beta", 0.10)
        lambda_value = getattr(self.args, "cma_lambda", 0.80)

        if beta <= 0.0 or beta > 1.0:
            raise ValueError(f"cma_beta must be in (0, 1], got {beta}")

        if lambda_value <= 0.0 or lambda_value > 1.0:
            raise ValueError(f"cma_lambda must be in (0, 1], got {lambda_value}")

        contribution_scores = {}

        for client_id in self.secure_server:
            ks_value = self.client_status[client_id].get("last_ks_value")
            pva_mean = self.client_status[client_id].get("last_pva_mean")

            if ks_value is None:
                ks_value = 0.0

            if pva_mean is None:
                # If MPDD is unavailable, use only MQV.
                pva_mean = 1.0

            ks_value = max(float(ks_value), 0.0)
            pva_mean = max(float(pva_mean), 0.0)

            contribution_scores[client_id] = ks_value * pva_mean

        sorted_clients = sorted(
            self.secure_server,
            key=lambda client_id: contribution_scores[client_id],
            reverse=True
        )

        num_secure_clients = len(sorted_clients)

        num_high_clients = max(
            1,
            int(np.ceil(num_secure_clients * beta))
        )

        high_clients = sorted_clients[:num_high_clients]
        low_clients = sorted_clients[num_high_clients:]

        aggregation_weights = {}

        # High-contribution group receives lambda total weight.
        high_weight_each = lambda_value / len(high_clients)

        for client_id in high_clients:
            aggregation_weights[client_id] = high_weight_each

        # Remaining clients share 1 - lambda total weight.
        if len(low_clients) > 0:
            low_weight_each = (1.0 - lambda_value) / len(low_clients)

            for client_id in low_clients:
                aggregation_weights[client_id] = low_weight_each

        else:
            # If there is no low-contribution group, high clients receive all weight.
            for client_id in high_clients:
                aggregation_weights[client_id] = 1.0 / len(high_clients)

        print("CMA aggregation setting:")
        print(f"  beta   = {beta:.2f}")
        print(f"  lambda = {lambda_value:.2f}")
        print(f"  high-contribution clients: {high_clients}")
        print(f"  low-contribution clients : {low_clients}")

        print("CMA contribution scores and aggregation weights:")

        for client_id in sorted_clients:
            print(
                f"  {self.clients[client_id]} "
                f"(index={client_id}) "
                f"score={contribution_scores[client_id]:.6f}, "
                f"weight={aggregation_weights[client_id]:.6f}, "
                f"KS={self.client_status[client_id].get('last_ks_value')}, "
                f"PVA={self.client_status[client_id].get('last_pva_mean')}"
            )

        return aggregation_weights

    def aggregation(self):
        """
        Aggregate secure client model weights.

        Supported methods:
        - fedavg: equal-weight FedAvg
        - cma: contribution-based model aggregation with beta and lambda

        CMA:
            beta = proportion of high-contribution clients
            lambda = total weight assigned to high-contribution group
        """
        if len(self.secure_server) == 0:
            raise RuntimeError("No secure clients available for aggregation.")

        aggregation_method = getattr(
            self.args,
            "aggregation_method",
            "cma"
        )

        if aggregation_method == "fedavg":
            aggregation_weights = {
                client_id: 1.0 / len(self.secure_server)
                for client_id in self.secure_server
            }

            print("Aggregation method: FedAvg")

        elif aggregation_method == "cma":
            aggregation_weights = self.compute_cma_group_weights()

            print("Aggregation method: CMA")

        else:
            raise ValueError(
                f"Unsupported aggregation_method: {aggregation_method}"
            )

        aggregated_weights = None

        for client_id in self.secure_server:
            client_weights = self.nns[client_id].get_weights()
            client_aggregation_weight = aggregation_weights[client_id]

            if aggregated_weights is None:
                aggregated_weights = [
                    layer_weight * client_aggregation_weight
                    for layer_weight in client_weights
                ]
            else:
                for layer_id in range(len(client_weights)):
                    aggregated_weights[layer_id] += (
                        client_weights[layer_id] * client_aggregation_weight
                    )

        self.nn.set_weights(aggregated_weights)

        print(
            f"Aggregated {len(self.secure_server)} client model(s): "
            f"{self.secure_server}"
        )

    def global_test(self):
        """
        Evaluate the final global model on all client datasets.

        All clients are tested, including warning and removed clients.
        """
        model = self.nn

        total_acc_all = 0.0
        total_acc_aggregated = 0.0

        all_count = 0
        aggregated_count = 0

        print("\nFinal global model testing on all clients:")
        print("-" * 150)
        print(
            f"{'Client':<12}"
            f"{'Index':<8}"
            f"{'Status':<12}"
            f"{'RemovedBy':<14}"
            f"{'Round':<8}"
            f"{'Test_ACC':<12}"
            f"{'Val_ACC':<12}"
            f"{'KS_value':<14}"
            f"{'PVA_mean':<14}"
            f"{'AbnCnt':<10}"
            f"{'LastReason':<16}"
            f"{'Aggregated':<12}"
        )
        print("-" * 150)

        for client_id in range(self.args.K):
            client_name = self.clients[client_id]
            model.file_name = client_name

            acc = test(self.args, model)
            self.client_status[client_id]["final_accuracy"] = float(acc)

            status = self.client_status[client_id]["status"]
            removed_by = self.client_status[client_id]["removed_by"]
            removed_round = self.client_status[client_id]["removed_round"]
            val_accuracy = self.client_status[client_id]["last_val_accuracy"]
            ks_value = self.client_status[client_id]["last_ks_value"]
            pva_mean = self.client_status[client_id]["last_pva_mean"]
            abnormal_count = self.client_status[client_id]["abnormal_count"]
            last_reason = self.client_status[client_id]["last_abnormal_reason"]
            used_for_aggregation = client_id in self.secure_server

            total_acc_all += acc
            all_count += 1

            if used_for_aggregation:
                total_acc_aggregated += acc
                aggregated_count += 1

            removed_by_text = removed_by if removed_by is not None else "-"
            removed_round_text = removed_round if removed_round is not None else "-"
            ks_text = f"{ks_value:.6f}" if ks_value is not None else "-"
            pva_text = f"{pva_mean:.6f}" if pva_mean is not None else "-"
            last_reason_text = last_reason if last_reason is not None else "-"
            aggregated_text = "Yes" if used_for_aggregation else "No"

            print(
                f"{client_name:<12}"
                f"{client_id:<8}"
                f"{status:<12}"
                f"{removed_by_text:<14}"
                f"{removed_round_text!s:<8}"
                f"{acc:<12.6f}"
                f"{ks_text:<14}"
                f"{pva_text:<14}"
                f"{abnormal_count:<10}"
                f"{last_reason_text:<16}"
                f"{aggregated_text:<12}"
            )

        print("-" * 150)

        avg_acc_all = total_acc_all / max(all_count, 1)
        avg_acc_aggregated = total_acc_aggregated / max(aggregated_count, 1)

        print(f"Average accuracy on all clients         : {avg_acc_all:.6f}")
        print(f"Average accuracy on aggregated clients  : {avg_acc_aggregated:.6f}")
        print(f"Candidate clients                       : {self.candidate_clients}")
        print(f"Secure aggregation clients              : {self.secure_server}")
        print(f"Permanently removed clients             : {self.removed_clients}")

        self.avg_acc_all = avg_acc_all
        self.avg_acc_retained = avg_acc_aggregated

        return avg_acc_all

    def save_client_status_csv(self, result_dir):
        """
        Save client-level final status to CSV.
        """
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        output_file = result_dir / "client_status_summary.csv"

        fieldnames = [
            "client_index",
            "client_name",
            "status",
            "removed_round",
            "removed_by",
            "last_ks_value",
            "last_val_accuracy",
            "last_pva_mean",
            "abnormal_count",
            "last_abnormal_reason",
            "final_accuracy",
            "used_for_aggregation",
            "in_candidate_clients",
            "in_secure_server",
            "in_removed_clients",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for client_id in range(self.args.K):
                info = self.client_status[client_id]

                writer.writerow({
                    "client_index": client_id,
                    "client_name": info["client_name"],
                    "status": info["status"],
                    "removed_round": info["removed_round"],
                    "removed_by": info["removed_by"],
                    "last_ks_value": info["last_ks_value"],
                    "last_val_accuracy": info["last_val_accuracy"],
                    "last_pva_mean": info["last_pva_mean"],
                    "abnormal_count": info["abnormal_count"],
                    "last_abnormal_reason": info["last_abnormal_reason"],
                    "final_accuracy": info["final_accuracy"],
                    "used_for_aggregation": client_id in self.secure_server,
                    "in_candidate_clients": client_id in self.candidate_clients,
                    "in_secure_server": client_id in self.secure_server,
                    "in_removed_clients": client_id in self.removed_clients,
                })

        print(f"Client status summary saved to: {output_file}")