import optuna
import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib

from itertools import product
from functools import partial
from .base import SearchStrategyBase

class brute_force(SearchStrategyBase):
    def _post_init_setup(self):
        self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
        self.metric_names = list(sorted(self.config["metrics"].keys()))
        if not self.sum_scaled_metrics:
            self.directions = [
                self.config["metrics"][k]["direction"] for k in self.metric_names
            ]
        else:
            self.direction = self.config["setup"]["direction"]

    
    def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
        # note that model can be mase_graph or nn.Module
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.sw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
        metrics = {}
        if is_eval_mode:
            with torch.no_grad():
                for runner in self.hw_runner:
                    metrics |= runner(self.data_module, model, sampled_config)
        else:
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
        return metrics

    def objective(self, search_space, sampled_config, model, is_eval_mode: bool):
        
        software_metrics = self.compute_software_metrics(
            model, sampled_config, is_eval_mode
        )
        hardware_metrics = self.compute_hardware_metrics(
            model, sampled_config, is_eval_mode
        )
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}

        

        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )

        if self.sum_scaled_metrics:
            a = sum(scaled_metrics.values())
            self.config["metrics"]['sum_scaled_metrics'] = {'scale': 1.0, 'direction': self.direction}
            scaled_metrics["sum_scaled_metrics"] = a
        
        return scaled_metrics

    def search(self, search_space) :
        
        bardia_values = list(search_space.choices_flattened.values())
        bardia_lengths = list(search_space.choice_lengths_flattened.values())
        
        bardia_indexes = []
        for i in bardia_lengths:
            bardia_indexes.append(list(range(i)))
    
        bardia_configs=[]
        for combination in product(*bardia_indexes):
            bardia_config = dict(zip(search_space.choices_flattened.keys(), combination))
            bardia_configs.append(bardia_config)

        perfs = []
        score = 0 
        for i in bardia_configs:
            sampled_config = search_space.flattened_indexes_to_config(i)

            is_eval_mode = self.config.get("eval_mode", True)
            model = search_space.rebuild_model(sampled_config, is_eval_mode)
            
            score = self.objective(search_space, sampled_config, model, is_eval_mode)
            perfs.append(sampled_config | score)
        
        perfs_df = pd.DataFrame(perfs)
        perfs_df.to_csv(self.save_dir / "perfs.csv")

        if(self.sum_scaled_metrics):
            metric_columns = ['sum_scaled_metrics']
        else:
            metric_columns = self.metric_names

        maximize = {metric: self.config["metrics"][metric]["direction"] == "maximize" for metric in metric_columns}
        best_df = find_pareto_optimal_df(perfs_df, metric_columns, maximize)
        
        best_df.to_csv(self.save_dir / "best_perf.csv")
        
        print(self.save_dir)
        print(best_df)

        # txt = "Best trial(s):\n"
        # df_truncated = best_df.loc[
        #     :, ["number", "software_metrics", "hardware_metrics", "scaled_metrics"]
        # ]

        # def beautify_metric(metric: dict):
        #     beautified = {}
        #     for k, v in metric.items():
        #         if isinstance(v, (float, int)):
        #             beautified[k] = round(v, 3)
        #         else:
        #             txt = str(v)
        #             if len(txt) > 20:
        #                 txt = txt[:20] + "..."
        #             else:
        #                 txt = txt[:20]
        #             beautified[k] = txt
        #     return beautified

        # df_truncated.loc[
        #     :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        # ] = df_truncated.loc[
        #     :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        # ].map(
        #     beautify_metric
        # )
        # txt += tabulate(
        #     df_truncated,
        #     headers="keys",
        #     tablefmt="orgtbl",
        # )
        # logger.info(f"Best trial(s):\n{txt}")
        return df
    
        return perfs_df

def find_pareto_optimal_df(df, metric_columns, maximize):
    """
    Find the Pareto optimal rows from a DataFrame based on specified metric columns.
    
    :param df: A pandas DataFrame.
    :param metric_columns: A list of column names representing the metrics.
    :param maximize: A boolean or a dictionary indicating if the metrics should be maximized or minimized.
                     If it's a boolean, it applies to all metrics. If it's a dictionary, it specifies
                     the behavior for each metric column.
    :return: A DataFrame with Pareto optimal rows.
    """
    if isinstance(maximize, bool):
        maximize = {metric: maximize for metric in metric_columns}

    def is_dominated(row, other_row):
        for metric in metric_columns:
            if (maximize[metric] and row[metric] < other_row[metric]) or (not maximize[metric] and row[metric] > other_row[metric]):
                return False
        return True

    pareto_optimal_indices = []
    for idx, row in df.iterrows():
        if not any(is_dominated(row, df.loc[other_idx]) for other_idx in df.index if other_idx != idx):
            pareto_optimal_indices.append(idx)
    
    return df.loc[pareto_optimal_indices]


        
    @staticmethod
    def _save_search_DataFrame(study: optuna.study.Study, search_space, save_path):
        df = study.trials_DataFrame(
            attrs=(
                "number",
                "value",
                "user_attrs",
                "system_attrs",
                "state",
                "datetime_start",
                "datetime_complete",
                "duration",
            )
        )
        df.to_json(save_path, orient="index", indent=4)
        return df

    @staticmethod
    def _save_best(study: optuna.study.Study, save_path):
        df = pd.DataFrame(
            columns=[
                "number",
                "value",
                "software_metrics",
                "hardware_metrics",
                "scaled_metrics",
                "sampled_config",
            ]
        )
        if study._is_multi_objective:
            best_trials = study.best_trials
            for trial in best_trials:
                row = [
                    trial.number,
                    trial.values,
                    trial.user_attrs["software_metrics"],
                    trial.user_attrs["hardware_metrics"],
                    trial.user_attrs["scaled_metrics"],
                    trial.user_attrs["sampled_config"],
                ]
                df.loc[len(df)] = row
        else:
            best_trial = study.best_trial
            row = [
                best_trial.number,
                best_trial.value,
                best_trial.user_attrs["software_metrics"],
                best_trial.user_attrs["hardware_metrics"],
                best_trial.user_attrs["scaled_metrics"],
                best_trial.user_attrs["sampled_config"],
            ]
            df.loc[len(df)] = row
        df.to_json(save_path, orient="index", indent=4)

        txt = "Best trial(s):\n"
        df_truncated = df.loc[
            :, ["number", "software_metrics", "hardware_metrics", "scaled_metrics"]
        ]

        def beautify_metric(metric: dict):
            beautified = {}
            for k, v in metric.items():
                if isinstance(v, (float, int)):
                    beautified[k] = round(v, 3)
                else:
                    txt = str(v)
                    if len(txt) > 20:
                        txt = txt[:20] + "..."
                    else:
                        txt = txt[:20]
                    beautified[k] = txt
            return beautified

        df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        ] = df_truncated.loc[
            :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
        ].map(
            beautify_metric
        )
        txt += tabulate(
            df_truncated,
            headers="keys",
            tablefmt="orgtbl",
        )
        logger.info(f"Best trial(s):\n{txt}")
        return df
