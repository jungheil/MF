# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import re
from functools import reduce
from typing import Callable, ClassVar, Dict, List, Optional, Protocol, Tuple

import mindspore as ms
import numpy as np
from mindspore import get_auto_parallel_context
from mindspore._c_expression import (
    disable_stress_test,
    enable_stress_test,
    # get_norm_print,
    get_commu_features,
)
from mindspore.communication.comm_func import all_gather_into_tensor, barrier
from mindspore.communication.management import get_group_size, get_rank, GlobalComm
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.train.serialization import _get_cur_rank_dp
from mindspore.train.stress_test import StressTestDataComparator, StressTestOnlineCB

from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister
from mindformers.tools.utils import get_real_rank, is_last_pipeline_stage
from mindformers.version_control import is_version_ge


class StressTestDataCollector(Protocol):
    ALL_STEPS_SUPPORT: ClassVar[bool]

    # return [step, value]
    def __call__(self, begin_step, end_step) -> Dict[str, List[np.ndarray]]:
        return {"": []}


class LocalComputationDataCollector:
    ALL_STEPS_SUPPORT: ClassVar[bool] = True

    def __init__(self):
        if get_auto_parallel_context("dump_local_norm_path"):
            self.dump_path = os.path.join(
                get_auto_parallel_context("dump_local_norm_path"),
                f"rank_{get_real_rank()}",
            )
        self.parsers = {}
        if is_version_ge(ms.__version__, "2.5.0"):
            self.parsers["local_loss"] = self._get_parser(
                r"local_loss_\w+\d+_(\d+)\.npy", 1
            )
            self.parsers["device_local_norm"] = self._get_parser(
                r"device_local_norm_\w+\d+_(\d+)\.npy", 1
            )
        else:
            self.parsers["local_loss"] = self._get_parser(r"(\d+)_local_loss\.npy", 1)
            self.parsers["device_local_norm"] = self._get_parser(
                r"(\d+)_device_local_norm\.npy", 1
            )

    def _get_parser(self, pattern, idx_group):
        pattern = re.compile(pattern)

        def get_single_data(file_list):
            file_matches = list(
                filter(
                    lambda x: x,
                    map(lambda x: re.match(pattern, x), file_list),
                )
            )
            file_matches = sorted(file_matches, key=lambda x: int(x.group(idx_group)))
            file_paths = [x.group(0) for x in file_matches if x]

            ret = []
            for path in file_paths:
                data = np.load(os.path.join(self.dump_path, path), allow_pickle=False)
                ret.append(data)
            return ret

        return get_single_data

    def _get_dump_data(self):
        file_list = os.listdir(self.dump_path)
        logger.debug(f"Stress Test Data Collector - Dump file list: {file_list}")
        if not file_list:
            return {}

        return {key: parser(file_list) for key, parser in self.parsers.items()}

    def __call__(self, begin_step, end_step) -> Dict[str, List[np.ndarray]]:
        dump_data = self._get_dump_data()
        if not dump_data:
            logger.error("Stress Test Data Collector - No dump data found.")
            return {}

        dump_loss = dump_data.get("local_loss", [])
        if any(loss is None for loss in dump_loss):
            logger.warning(
                "Stress Test Data Collector - Incomplete local loss data found."
            )
            dump_loss = []
        if len(dump_loss) != end_step - begin_step:
            logger.warning(
                "Stress Test Data Collector - Local loss data length mismatch."
            )
            dump_loss = []

        dump_norm = dump_data.get("device_local_norm", [])
        if any(norm is None for norm in dump_norm):
            logger.warning(
                "Stress Test Data Collector - Incomplete device local norm data found."
            )
            dump_norm = []
        if len(dump_norm) != end_step - begin_step:
            logger.warning(
                "Stress Test Data Collector - Device local norm data length mismatch."
            )
            dump_norm = []

        ret = {}
        if dump_loss:
            ret["local_forward"] = dump_loss
        if dump_norm:
            ret["local_backward"] = dump_norm
        return ret


class CommOpsNewCollector:
    ALL_STEPS_SUPPORT: ClassVar[bool] = False

    def _parse_ops(self, ops_norm):
        f_send_list = []
        b_send_list = []
        comm_list = []

        ops_norm = sorted(ops_norm.items(), key=lambda x: int(x[0].split("|")[0]))

        send_pattern = re.compile(r"^[\w/-]+/Send-\w*$")
        for k, v in ops_norm:
            ops_name = k.split("|")[1]
            # ops_group_str = k.split("|")[2]
            # if ops_group_str == "no_value":
            #     raise ValueError(f"Invalid ops group in ops norm {k}.")
            # ops_group = list(map(int, ops_group_str.split(",")))
            v = np.array(v)
            if send_pattern.match(ops_name):
                head = ops_name.split("/")[0]
                if head == "Gradients":
                    b_send_list.append(v)
                else:
                    f_send_list.append(v)
            else:
                comm_list.append(v)
        return f_send_list, b_send_list, comm_list

    def __call__(self, begin_step, end_step) -> Dict[str, List[np.ndarray]]:
        ret = {}
        ops_norm = get_commu_features()
        f_send_list, b_send_list, comm_list = self._parse_ops(ops_norm)
        ret["send_ops_forward"] = [np.array(f_send_list)] if f_send_list else []
        ret["send_ops_backward"] = [np.array(b_send_list)] if b_send_list else []
        ret["comm_ops"] = [np.array(comm_list)] if comm_list else []
        return ret


class TensorDumpCommOpsCollector:
    ALL_STEPS_SUPPORT: ClassVar[bool] = True

    def __init__(self, dump_path):
        self.tensor_dump_path = dump_path
        self.cur_rank = get_real_rank()

        for dirpath, dirs, files in os.walk(self.tensor_dump_path):
            for file in files:
                if file == "statistic.csv":
                    try:
                        os.remove(os.path.join(dirpath, file))
                    except Exception as e:
                        logger.error(f"Error removing {file}: {e}")

    def __call__(self, begin_step, end_step) -> Dict[str, List[np.ndarray]]:
        ret = {}
        f_send_list = []
        b_send_list = []
        comm_list = []
        for i in range(begin_step, end_step):
            comm_ops = self._get_tensor_dump(
                self.tensor_dump_path,
                "net",
                "1",
                i - 1,
                self.cur_rank,
            )
            f_send, b_send, comm = self._parse_tensor_dump_ops(comm_ops)

            f_send_list.append(f_send)
            b_send_list.append(b_send)
            comm_list.append(comm)

        ret["send_ops_forward"] = f_send_list
        ret["send_ops_backward"] = b_send_list
        ret["comm_ops"] = comm_list
        return ret

    def _get_tensor_dump(self, dump_path, net_name, graph_id, iter_id, rank_id):
        path = os.path.join(
            dump_path,
            f"rank_{rank_id}",
            net_name,
            str(graph_id),
            str(iter_id),
            "statistic.csv",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stress Test: tensor dump file {path} not found.")
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(f"Stress Test: tensor dump file {path} is empty.")

        def _parse_to_value(value):
            return float(value)

        def _parse_md5(value):
            return sum([ord(c) for c in value]) / (
                reduce(lambda x, y: x ^ y, [ord(c) for c in value]) + 1e-6
            )
            # return sum([ord(c) for c in value])

        header = lines[0].strip().split(",")
        value_col_idx = -1
        if "MD5" in header:
            value_col_idx = header.index("MD5")
            parser = _parse_md5
        elif "L2Norm Value" in header:
            value_col_idx = header.index("L2Norm Value")
            parser = _parse_to_value
        else:
            raise NotImplementedError(
                f"Stress Test: tensor dump file {path} has no MD5 or L2Norm column."
            )
        type_idx = header.index("Op Type")
        name_idx = header.index("Op Name")
        io_idx = header.index("IO")
        tensor_dump = []
        for line in lines[1:]:
            # parts = line.strip().split(",")
            parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line.strip())
            # if parts[io_idx] != "input":
            tensor_type = parts[type_idx].strip()
            tensor_name = parts[name_idx].strip()
            tensor_io = parts[io_idx].strip()
            tensor_value = parts[value_col_idx].strip()

            if not tensor_value:
                raise ValueError(
                    f"Stress Test: tensor {tensor_name} in file {path} has no value."
                )
            tensor_dump.append(
                (tensor_type, tensor_name, tensor_io, parser(tensor_value))
            )
        return tensor_dump

    def _parse_tensor_dump_ops(self, comm_ops):
        f_send_ops = []
        b_send_ops = []
        comm_ops = []

        for op_type, name, io, value in comm_ops:
            if io != "input":
                continue
            if op_type == "Send":
                head = name.split("/")[0]
                if head == "Gradients":
                    b_send_ops.append(value)
                else:
                    f_send_ops.append(value)
            else:
                comm_ops.append(value)
        return f_send_ops, b_send_ops, comm_ops


def all_gather_func(tensor: np.ndarray, group: Optional[str] = None) -> np.ndarray:
    # convert numpy array to MindSpore tensor and add a leading dimension
    if group is None:
        group = GlobalComm.WORLD_COMM_GROUP
    tensor = ms.Tensor(np.expand_dims(tensor, 0), dtype=ms.float32)
    gather_tensor = all_gather_into_tensor(tensor, group=group)
    barrier(group=group)
    gather_tensor = gather_tensor[0]
    return gather_tensor.asnumpy()


def get_dp_group(dp_size):
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    tp_size = per_stage_device_nums // dp_size
    tp_id = rank % tp_size
    rank_list = list(
        range(
            (stage_nums - 1) * per_stage_device_nums + tp_id,
            stage_nums * per_stage_device_nums + tp_id,
            tp_size,
        )
    )
    return rank_list


def get_pipeline_group():
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    local_stage_rank_id = rank % per_stage_device_nums
    rank_list = [
        local_stage_rank_id + x * per_stage_device_nums for x in range(0, stage_nums)
    ]
    return rank_list


def get_stage_group():
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_idx = rank // per_stage_device_nums
    stage_rank = list(
        range(
            stage_idx * per_stage_device_nums,
            (stage_idx + 1) * per_stage_device_nums,
        )
    )
    return stage_rank


def get_rank_stage():
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage = rank // per_stage_device_nums
    return stage


@MindFormerRegister.register(MindFormerModuleType.CALLBACK)
class MFStressTestCB(StressTestOnlineCB):
    def __init__(self, data_parallel, micro_batch, stress_test_steps):
        super().__init__()
        self.stress_test_ctx.set_data_queue_config(data_parallel, micro_batch)
        self.stress_test_steps: List[int] = stress_test_steps

        self.local_data_collector: StressTestDataCollector = (
            LocalComputationDataCollector()
        )
        self.comm_ops_collector: StressTestDataCollector = CommOpsNewCollector()
        self.comparator: StressTestDataComparator = StressTestDataComparator(
            get_rank(),
            get_dp_group(data_parallel),
            auto_parallel_context().get_pipeline_stages() != 1,
            get_rank_stage(),
            is_last_pipeline_stage(),
            get_pipeline_group(),
            get_stage_group(),
            all_gather_func,
            logger,
        )

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        self.cur_step = cb_params.cur_step_num

        if self.stress_test_ctx.in_stress_test:
            local_data = self.local_data_collector(
                self.cur_step - self.stress_test_ctx.sink_size + 1, self.cur_step + 1
            )
            comm_ops_data = self.comm_ops_collector(
                self.cur_step - self.stress_test_ctx.sink_size + 1, self.cur_step + 1
            )

            if (
                not self.local_data_collector.ALL_STEPS_SUPPORT
                or not self.comm_ops_collector.ALL_STEPS_SUPPORT
            ):
                if self.local_data_collector.ALL_STEPS_SUPPORT:
                    local_data = {k: v[-1:] for k, v in local_data.items()}
                if self.comm_ops_collector.ALL_STEPS_SUPPORT:
                    comm_ops_data = {k: v[-1:] for k, v in comm_ops_data.items()}
            data = {**local_data, **comm_ops_data}
            data = {k: np.array(v) for k, v in data.items()}
            print(data)
            result = self.comparator(**data)
            print(result)

        super().on_train_step_end(run_context)
