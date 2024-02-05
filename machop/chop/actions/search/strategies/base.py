import math
import joblib

from .runners import get_sw_runner, get_hw_runner


class SearchStrategyBase:
    """
    Base class for search strategies.

    ---

    What is a search strategy?

    search_strategy is responsible for:
    - setting up the data loader
    - perform search, which includes:
        - sample the search_space.choice_lengths_flattened to get the indexes
        - call search_space.rebuild_model to build a new model with the sampled indexes
        - calculate the software & hardware metrics through the sw_runners and hw_runners
    - save the results

    ---
    To implement a new search strategy, you need to implement the following methods:
    - `_post_init_setup(self) -> None`: additional setup for the subclass instance
    - `search(self, search_space) -> Any: perform search, and save the results

    ---
    Check `machop/chop/actions/search/strategies/optuna.py` for an example.
    """

    is_iterative: bool = None  # whether the search strategy is iterative or not

    def __init__(
        self,
        model_info,
        data_module,
        dataset_info,
        task: str,
        config: dict,
        accelerator,
        save_dir,
        visualizer,
    ):
        self.dataset_info = dataset_info
        self.task = task
        self.config = config
        self.accelerator = accelerator
        self.save_dir = save_dir
        self.data_module = data_module
        self.visualizer = visualizer

        self.sw_runner = []
        self.hw_runner = []
        # the software runner's __call__ will use the rebuilt model to calculate the software metrics like accuracy, loss, ...

        for runner_name, runner_cfg in config["sw_runner"].items():
            self.sw_runner.append(
                get_sw_runner(
                    runner_name, model_info, task, dataset_info, accelerator, runner_cfg
                )
            )
        # the hardware runner's __call__ will use the rebuilt model to calculate the hardware metrics like average bitwidth, latency, ...
        for runner_name, runner_cfg in config["hw_runner"].items():
            self.hw_runner.append(
                get_hw_runner(
                    runner_name, model_info, task, dataset_info, accelerator, runner_cfg
                )
            )

        self._post_init_setup()

    @staticmethod
    def _save_study(study, save_path):
        """
        Save the study object. The subclass can call this method to save the study object at the end of the search.
        """
        with open(save_path, "wb") as f:
            joblib.dump(study, f)

    def _post_init_setup(self):
        """
        Post init setup. This is where additional config parsing and setup should be done for the subclass instance.
        """
        raise NotImplementedError()

    def search(self, search_space):
        """
        Perform search, and save the results.
        """
        raise NotImplementedError()

class bardia_exhaustive_search(SearchStrategyBase):
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

    def objective(self, search_space):
        sampled_indexes = {}

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

        if not self.sum_scaled_metrics:
            return list(scaled_metrics.values())
        else:
            return sum(scaled_metrics.values())

    def search(self, search_space) -> optuna.study.Study:
        
        if not self.sum_scaled_metrics:
            study_kwargs["directions"] = self.directions
        else:
            study_kwargs["direction"] = self.direction

        a = search_space.choice_lengths_flattened.items()
            # sampled_indexes[name] = trial.suggest_int(name, 0, length - 1)
        sampled_config = search_space.flattened_indexes_to_config(sampled_indexes)

        is_eval_mode = self.config.get("eval_mode", True)
        model = search_space.rebuild_model(sampled_config, is_eval_mode)
        # if isinstance(self.config["setup"].get("pkl_ckpt", None), str):
        #     study = joblib.load(self.config["setup"]["pkl_ckpt"])
        #     logger.info(f"Loaded study from {self.config['setup']['pkl_ckpt']}")
        # else:
        #     study = optuna.create_study(**study_kwargs)

        study.optimize(
            func=partial(self.objective, search_space=search_space),
            n_jobs=self.config["setup"]["n_jobs"],
            n_trials=self.config["setup"]["n_trials"],
            timeout=self.config["setup"]["timeout"],
            callbacks=[
                partial(
                    callback_save_study,
                    save_dir=self.save_dir,
                    save_every_n_trials=self.config["setup"].get(
                        "save_every_n_trials", 10
                    ),
                )
            ],
            show_progress_bar=True,
        )
        self._save_study(study, self.save_dir / "study.pkl")
        self._save_search_dataframe(study, search_space, self.save_dir / "log.json")
        self._save_best(study, self.save_dir / "best.json")

        for i in search_space.choices_flattened:
            best_model 
        return study

    @staticmethod
    def _save_search_dataframe(study: optuna.study.Study, search_space, save_path):
        df = study.trials_dataframe(
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
