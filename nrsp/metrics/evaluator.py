from abc import ABC, abstractmethod
import csv
import os
from collections import defaultdict
import numpy.typing as npt

from nrsp.metrics.cfar_metrics import calc_metrics, calc_metrics_extended


class Evaluator(ABC):

    @abstractmethod
    def eval():
        pass

    @abstractmethod
    def results():
        pass

    @abstractmethod
    def save():
        pass


class CfarEvaluator(Evaluator):

    def __init__(self, keys: list[str], detection_area):
        """TODO

        :param list[str] keys: _description_
        :param _type_ detection_area: _description_
        """

        self.detection_area = detection_area
        self._keys = keys

        # indexed by _get_key(...)
        self._results = dict()
        self._n = defaultdict(int)

    def _get_key(self, **kwargs) -> str:
        """Generate unique keyword string from kwargs.
        Only keys as listed in self._keys are taken into account.

        :return str: uniqe key from given kwargs
        """

        key = ""
        for k in self._keys:
            key += f"{k}={kwargs.get(k)};"
        return key

    def _default_result(self, **kwargs) -> dict:
        """Generate initial result dict. Metrics are initialized to zero and all keys
        in self._keys are added. If key values any are present in kwargs, initialize to this value.

        :return dict: initialized result dict
        """

        r = {
            "recall": 0,
            "precision": 0,
            "recall_ext": 0,
            "precision_ext": 0,
            "iou_score_ext": 0,
        }
        for k in self._keys:
            r[k] = None
        r.update(kwargs)
        return r

    def eval(
        self,
        cfar_map: npt.NDArray,
        targets_map: npt.NDArray,
        targets_ext: npt.NDArray = None,
        **kwargs,
    ):
        """Calculate metrics from the cfar map and the targets map and
        accumulate them for the specified kwargs.
        targets_ext may be used as to not compute the extented target map
        in every metrics.calc_metrics_extended(...) call.

        :param npt.NDArray cfar_map: cfar map
        :param npt.NDArray targets_map: targets map
        :param npt.NDArray targets_ext: targets map with applied detection area, defaults to None
        """

        recall, precision = calc_metrics(cfar_map, targets_map)
        recall_ext, precision_ext, iou_score_ext = calc_metrics_extended(cfar_map, targets_map, self.detection_area, targets_ext)

        key = self._get_key(**kwargs)
        if not self._results.get(key):
            self._results[key] = self._default_result(**kwargs)

        self._results[key]["key"] = key
        self._results[key]["recall"] += recall
        self._results[key]["precision"] += precision
        self._results[key]["recall_ext"] += recall_ext
        self._results[key]["precision_ext"] += precision_ext
        self._results[key]["iou_score_ext"] += iou_score_ext
        self._n[key] += 1

    def results(self) -> list[dict]:
        """Return list of metric results for all parameters.

        :return list[dict]: list of metric results
        """

        res = []
        for _, v in self._results.items():
            r = {}

            # add keys and detection_area
            for k in self._keys:
                r[k] = v[k]
            r["detection_area"] = self.detection_area

            # add metrics
            r["recall"] = v["recall"] / self._n[v["key"]]
            r["precision"] = v["precision"] / self._n[v["key"]]
            r["recall_ext"] = v["recall_ext"] / self._n[v["key"]]
            r["precision_ext"] = v["precision_ext"] / self._n[v["key"]]
            r["iou_score_ext"] = v["iou_score_ext"] / self._n[v["key"]]

            f_score = 2 * (r["precision"] * r["recall"]) / (r["precision"] + r["recall"] + 1e-8)
            f_score_ext = 2 * (r["precision_ext"] * r["recall_ext"]) / (r["precision_ext"] + r["recall_ext"] + 1e-8)
            r["f_score"] = f_score
            r["f_score_ext"] = f_score_ext

            res.append(r)

        return res

    def save(self, filename: str, global_params: dict = None):
        """Save results in csv. Add global_params to every entry

        :param str filename: output file name/path
        :param dict global_params: global evaluation params, defaults to None
        """

        # check if file already exists and create directory if necessary
        file_exists = os.path.isfile(filename)
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        keys = list(global_params.keys())
        keys += self._keys
        keys += ["detection_area", "recall", "precision", "f_score", "recall_ext", "precision_ext", "f_score_ext", "iou_score_ext"]

        with open(filename, "a", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)

            if not file_exists:
                dict_writer.writeheader()

            for r in self.results():
                r.update(global_params)
                dict_writer.writerow(r)


class CfarTurnoffEvaluator(CfarEvaluator):
    """CFAR Evaluator that also tracks average neuron turnoff percentage."""

    def _default_result(self, **kwargs) -> dict:
        """Generate initial result dict. Metrics are initialized to zero and all keys
        in self._keys are added. If key values any are present in kwargs, initialize to this value.

        :return dict: initialized result dict
        """
        r = super()._default_result(**kwargs)
        r["inactive_percentage"] = 0
        return r

    def eval(
        self,
        cfar_map: npt.NDArray,
        targets_map: npt.NDArray,
        targets_ext: npt.NDArray = None,
        inactive_percentage: float = 0.0,
        **kwargs,
    ):
        """Calculate metrics from the cfar map and the targets map and
        accumulate them for the specified kwargs.
        targets_ext may be used as to not compute the extented target map
        in every metrics.calc_metrics_extended(...) call.

        :param npt.NDArray cfar_map: cfar map
        :param npt.NDArray targets_map: targets map
        :param npt.NDArray targets_ext: targets map with applied detection area, defaults to None
        :param float inactive_percentage: neuron inactive percentage, defaults to 0.0
        """
        # Call parent eval method
        super().eval(cfar_map, targets_map, targets_ext, **kwargs)

        # Add inactive percentage tracking
        key = self._get_key(**kwargs)
        self._results[key]["inactive_percentage"] += inactive_percentage

    def results(self) -> list[dict]:
        """Return list of metric results for all parameters.

        :return list[dict]: list of metric results
        """
        res = super().results()

        # Add average turnoff percentage to each result
        for i, (_, v) in enumerate(self._results.items()):
            res[i]["inactive_percentage"] = v["inactive_percentage"] / self._n[v["key"]]

        return res

    def save(self, filename: str, global_params: dict = None):
        """Save results in csv. Add global_params to every entry

        :param str filename: output file name/path
        :param dict global_params: global evaluation params, defaults to None
        """
        # check if file already exists and create directory if necessary
        file_exists = os.path.isfile(filename)
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        keys = list(global_params.keys()) if global_params else []
        keys += self._keys
        keys += [
            "detection_area",
            "recall",
            "precision",
            "f_score",
            "recall_ext",
            "precision_ext",
            "f_score_ext",
            "iou_score_ext",
            "inactive_percentage",
        ]

        with open(filename, "a", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)

            if not file_exists:
                dict_writer.writeheader()

            for r in self.results():
                if global_params:
                    r.update(global_params)
                dict_writer.writerow(r)
