我实现了一个网络级别的压测工具，能够对特定训练网络、特性数据进行压测，以找出或复现软件、硬件存在的问题，如静默故障sdc。其中mindspore是神经网络框架，负责压测环境构建、3D比对算法（tp、dp、pp）定位到卡的能力，ctx是管理压测的，可以通过环境变量打开，然后llm框架mindformer可以通过ctx对压测使能、自定义在什么时候、那个step进行压测，什么时候进行恢复。现在我需要你帮我画4+1视图，表示mf、ms之间各个组件之间的关系。mindspore中压测相关代码如下：
# Copyright 2025 Huawei Technologies Co., Ltd
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
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, IntFlag
from functools import wraps
from typing import Callable, Dict, List, Optional

import numpy as np

import mindspore as ms
from mindspore import log as logger
from mindspore.communication.management import create_group
from mindspore.dataset.engine.datasets import _reset_training_dataset
from mindspore.train.callback import Callback


class StressTestMode(Enum):
    """Stress Test Mode Enum"""

    DISABLE = 0
    OFFLINE = 1
    ONLINE = 2


class StressTestRestartProcess(IntFlag):
    IDLE = 0
    BACKUP = 1
    PREPROCESS = 2
    RESTORE = 4


class StressTestDPCompareStatus(Enum):
    ALL_PASS = 0
    MAJORITY_AGREE = 1
    MAJORITY_DISAGREE = 2
    UNKNOWN = 3
    CANNOT_DETERMINE = 4


@dataclass
class StressTestResult:
    dp_compare_forward_match: Optional[List[Dict]] = None
    dp_compare_forward_mismatch: Optional[List[Dict]] = None
    dp_compare_backward_match: Optional[List[Dict]] = None
    dp_compare_backward_mismatch: Optional[List[Dict]] = None
    dp_status_forward: Optional[List[StressTestDPCompareStatus]] = None
    dp_status_backward: Optional[List[StressTestDPCompareStatus]] = None
    dp_status: Optional[List[StressTestDPCompareStatus]] = None
    pp_status: Optional[List[StressTestDPCompareStatus]] = None
    compare_rank_idx: Optional[List[int]] = None
    failed_stage: Optional[List[int]] = None
    failed_rank_idx: Optional[List[int]] = None


@dataclass
class StressTestData:
    local_forward: Optional[np.ndarray]
    local_backward: Optional[np.ndarray]
    comm_ops: Optional[np.ndarray]
    send_ops_forward: Optional[np.ndarray]
    send_ops_backward: Optional[np.ndarray]
    gather_local_forward: Optional[np.ndarray]
    gather_local_backward: Optional[np.ndarray]
    gather_comm_ops: Optional[np.ndarray]
    gather_send_forward: Optional[np.ndarray]
    gather_send_backward: Optional[np.ndarray]


def enable_data_queue_processing(data_parallel, micro_batch):
    """Enable data queue processing for stress test."""
    from mindspore._c_expression import enable_stress_test

    enable_stress_test(list(range(1, 10000)), data_parallel, micro_batch)


def disable_data_queue_processing():
    """Enable data queue processing for stress test."""
    from mindspore._c_expression import disable_stress_test

    disable_stress_test()


def _restore_dataset(step, train_dataset, batch_num, sink_mode, sink_size):
    """
    Restore training step.
    """
    if step < 0:
        raise RuntimeError("NeoStressTest: Invalid step to restore.")

    initial_epoch = step // batch_num
    initial_step = step % batch_num

    if sink_mode:
        train_dataset.set_init_step(initial_epoch)
        if sink_size > 0:
            initial_step = initial_epoch * sink_size + initial_step
        else:
            initial_step = (
                initial_epoch * train_dataset.get_dataset_size() + initial_step
            )
    else:
        train_dataset.set_init_step(initial_step)

    if hasattr(train_dataset, "_dataset_helper"):
        dataset_helper = train_dataset._dataset_helper
        _reset_training_dataset(
            initial_step, dataset_helper.iter.dataset.get_dataset_size()
        )


def _backup_parameters(parameters_and_names, stress_test_ctx):
    """
    Backup model parameters.
    """
    for k, v in parameters_and_names:
        stress_test_ctx._parameter_backup[k] = ms.Tensor(shape=v.shape, dtype=v.dtype)
        stress_test_ctx._parameter_backup[k].copy_(v)


def _restore_parameters(parameters_and_names, stress_test_ctx):
    """
    Restore model parameters from backup.
    """
    for k, v in parameters_and_names:
        if k in stress_test_ctx._parameter_backup:
            v.copy_(stress_test_ctx._parameter_backup[k])
    stress_test_ctx._parameter_backup.clear()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class StressTestCtx(metaclass=Singleton):
    """StressTest Singleton Class"""

    def __init__(self) -> None:
        self.test_mode = StressTestMode(int(os.getenv("MS_STRESSTEST", "0")))
        self._enable = self.test_mode != StressTestMode.DISABLE

        self.data_parallel = None
        self.micro_batch = None
        self.sink_mode = None
        self.sink_size = None

        # runtime parameters
        self.test_begin_step: Optional[int] = None
        self.in_stress_test = False
        self._restart_process = StressTestRestartProcess.IDLE
        self._parameter_backup = {}

    def set_data_queue_config(self, data_parallel, micro_batch):
        """Set data queue parameters for stress test."""
        # TODO Check input
        self.data_parallel = data_parallel
        self.micro_batch = micro_batch

        self._avaliable = True

    @property
    def restart_process(self):
        """Get restart process."""
        return self._restart_process

    @restart_process.setter
    def restart_process(self, process):
        # todo check process
        self._restart_process = process

    def is_enable(self):
        return self._enable

    def is_avaliable(self):
        """Check if stress test is avaliable."""
        return (
            self._enable
            and self.data_parallel is not None
            and self.micro_batch is not None
            and self.sink_mode is not None
            and self.sink_size is not None
        )


stress_test_ctx = StressTestCtx()


def stress_test_preprocess(test_begin_step, train_dataset, batch_num, stress_test_ctx):
    if not stress_test_ctx.is_avaliable():
        raise RuntimeError("StressTest is not avaliable.")
    enable_data_queue_processing(
        stress_test_ctx.data_parallel, stress_test_ctx.micro_batch
    )
    # XXX刷新数据集是否能被感知
    _restore_dataset(
        test_begin_step,
        train_dataset,
        batch_num,
        stress_test_ctx.sink_mode,
        stress_test_ctx.sink_size,
    )


def stress_test_backup(parameters_and_names, stress_test_ctx):
    """
    Initialization before stress test.
    """

    _backup_parameters(parameters_and_names, stress_test_ctx)


def stress_test_restore(
    test_begin_step,
    train_dataset,
    batch_num,
    parameters_and_names,
    stress_test_ctx,
):
    """
    Restorey after stress test.
    """
    logger.info("StressTest - Restoring from stress test...")

    disable_data_queue_processing()
    _restore_dataset(
        test_begin_step,
        train_dataset,
        batch_num,
        stress_test_ctx.sink_mode,
        stress_test_ctx.sink_size,
    )
    if stress_test_ctx._parameter_backup:
        _restore_parameters(parameters_and_names, stress_test_ctx)


class StressTestOnlineCB(Callback):
    # TODO step 1 not support
    def __init__(self):
        super().__init__()
        self.stress_test_ctx: StressTestCtx = StressTestCtx()
        if not self.stress_test_ctx.is_enable():
            raise RuntimeError(
                "StressTest - StressTestOnlineCB is enabled but StressTestCtx is not enable."
            )
        self.stress_test_steps = []

    def on_train_step_end(self, run_context):
        """
        Restore stress test at the end of step if needed.
        """
        if self.stress_test_ctx.test_mode == StressTestMode.ONLINE:
            self._handle_online(run_context)

    def _handle_online(self, run_context):
        cb_params = run_context.original_args()
        sink_end_step = cb_params.cur_step_num

        next_step = sink_end_step + 1
        if next_step in self.stress_test_steps:
            for step in range(next_step, next_step + self.stress_test_ctx.sink_size):
                if step not in self.stress_test_steps:
                    raise RuntimeError(
                        f"StressTest -  Step {step} status differs from start step {next_step} in sink with size {self.stress_test_ctx.sink_size}."
                    )
            if self.stress_test_ctx.in_stress_test:
                for step in range(
                    sink_end_step + 1 - self.stress_test_ctx.sink_size,
                    sink_end_step + 1,
                ):
                    self.stress_test_steps.remove(step)
            else:
                self.stress_test_ctx.test_begin_step = next_step
                run_context.request_stop()
                self.stress_test_ctx.restart_process = (
                    StressTestRestartProcess.BACKUP
                    | StressTestRestartProcess.PREPROCESS
                )
        else:
            if self.stress_test_ctx.in_stress_test:
                for step in range(
                    sink_end_step + 1 - self.stress_test_ctx.sink_size,
                    sink_end_step + 1,
                ):
                    self.stress_test_steps.remove(step)
                run_context.request_stop()
                self.stress_test_ctx.restart_process = StressTestRestartProcess.RESTORE


def _handle_stress_test(func):
    stress_test_ctx = StressTestCtx()
    if not stress_test_ctx.is_enable():
        return func

    @wraps(func)
    def offline_wrapper(self, *args, **kwargs):
        stress_test_ctx.test_begin_step = 1
        stress_test_ctx.sink_mode = (
            args[3] if len(args) > 3 else kwargs.get("dataset_sink_mode", True)
        )
        stress_test_ctx.sink_size = (
            (args[4] if len(args) > 4 else kwargs.get("sink_size", 1))
            if stress_test_ctx.sink_mode
            else 1
        )

        if not stress_test_ctx.is_enable() and not stress_test_ctx.is_avaliable():
            raise RuntimeError("StressTest is not avaliable.")
        train_dataset = args[1]
        stress_test_preprocess(
            stress_test_ctx.test_begin_step - 1,
            train_dataset,
            self.batch_num,
            stress_test_ctx,
        )
        stress_test_ctx.in_stress_test = True
        ret = func(self, *args, **kwargs)
        return ret

    @wraps(func)
    def online_wrapper(self, *args, **kwargs):
        stress_test_ctx.sink_mode = (
            args[3] if len(args) > 3 else kwargs.get("dataset_sink_mode", True)
        )
        stress_test_ctx.sink_size = (
            (args[4] if len(args) > 4 else kwargs.get("sink_size", 1))
            if stress_test_ctx.sink_mode
            else 1
        )
        while True:
            ret = func(self, *args, **kwargs)
            if stress_test_ctx.restart_process == StressTestRestartProcess.IDLE:
                return ret
            if stress_test_ctx.restart_process & StressTestRestartProcess.BACKUP:
                if not stress_test_ctx.is_avaliable():
                    raise RuntimeError("StressTest is not avaliable.")
                parameters_and_names = self._train_network.parameters_and_names()
                stress_test_backup(parameters_and_names, stress_test_ctx)
            if stress_test_ctx.restart_process & StressTestRestartProcess.PREPROCESS:
                if not stress_test_ctx.is_avaliable():
                    raise RuntimeError("StressTest is not avaliable.")
                train_dataset = args[1]
                stress_test_preprocess(
                    stress_test_ctx.test_begin_step - 1,
                    train_dataset,
                    self.batch_num,
                    stress_test_ctx,
                )
                kwargs["initial_step"] = stress_test_ctx.test_begin_step - 1
                stress_test_ctx.in_stress_test = True

            if stress_test_ctx.restart_process & StressTestRestartProcess.RESTORE:
                logger.info("StressTest - Restoring from stress test...")
                train_dataset = args[1]
                parameters_and_names = self._train_network.parameters_and_names()
                stress_test_restore(
                    stress_test_ctx.test_begin_step - 1,
                    train_dataset,
                    self.batch_num,
                    parameters_and_names,
                    stress_test_ctx,
                )
                kwargs["initial_step"] = stress_test_ctx.test_begin_step - 1
                stress_test_ctx.in_stress_test = False

            stress_test_ctx.restart_process = StressTestRestartProcess.IDLE

    if stress_test_ctx.test_mode == StressTestMode.OFFLINE:
        logger.info("StressTest - Trainer wrapped with offline stress test.")
        return offline_wrapper
    elif stress_test_ctx.test_mode == StressTestMode.ONLINE:
        logger.info("StressTest - Trainer wrapped with online stress test.")
        return online_wrapper
    else:
        raise RuntimeError("Unsupported StressTest mode.")


class StressTestDataComparator:
    def __init__(
        self,
        cur_rank,
        dp_domain_ranks,
        is_pipeline_parallel,
        cur_stage,
        is_last_stage,
        pp_group_ranks,
        stage_group_ranks,
        all_gather_func,
        logger=logger,
    ):
        self.cur_rank: int = cur_rank
        self.cur_stage: int = cur_stage
        self.is_pipeline_parallel: bool = is_pipeline_parallel
        self.is_last_stage: bool = is_last_stage
        self.dp_domain_ranks: List[int] = dp_domain_ranks
        self.pp_group_ranks: List[int] = pp_group_ranks
        self.stage_group_ranks: List[int] = stage_group_ranks
        self.all_gather_func: Callable[[np.ndarray, Optional[str]], np.ndarray] = (
            all_gather_func
        )
        self.logger = logger

        # TODO Check
        if len(dp_domain_ranks) < 2:
            raise ValueError("StressTest - Data parallel domain ranks less than 2.")

        self.dp_group = "-".join(map(str, self.dp_domain_ranks))
        self.stage_group = None
        create_group(self.dp_group, self.dp_domain_ranks)
        if is_pipeline_parallel:
            self.pp_group = "-".join(map(str, self.pp_group_ranks))
            create_group(self.pp_group, self.pp_group_ranks)
            self.stage_group = "-".join(map(str, self.stage_group_ranks))
            create_group(self.stage_group, self.stage_group_ranks)

    def __call__(
        self,
        local_forward: np.ndarray,
        local_backward: np.ndarray,
        comm_ops: np.ndarray,
        send_ops_forward: np.ndarray,
        send_ops_backward: np.ndarray,
    ):
        data = self.get_data(
            local_forward,
            local_backward,
            comm_ops,
            send_ops_forward,
            send_ops_backward,
        )
        result = StressTestResult()
        # dp failures
        result = self.analyze_dp_failures(data, result)

        if self.is_pipeline_parallel:
            result = self.get_pp_compare_rank(result)

        if len(self.dp_domain_ranks) == 2:
            result.compare_rank_idx = (
                [0] * len(result.dp_status)
                if self.dp_domain_ranks.index(self.cur_rank)
                else [1] * len(result.dp_status)
            )

        # TODO 早返回， pp status 和 dp status pass or cannot
        if all(s == StressTestDPCompareStatus.ALL_PASS for s in result.dp_status) and (
            all(s == StressTestDPCompareStatus.ALL_PASS for s in result.pp_status)
            if self.is_pipeline_parallel
            else True
        ):
            self.logger.info("StressTest - No failures detected.")
            return result

        if self.is_pipeline_parallel:
            result = self.analyze_stage_failures(data, result)

        result = self.analyze_tp_failures(data, result)

        self.logger.info(result)

    def get_data(
        self,
        local_forward: np.ndarray,
        local_backward: np.ndarray,
        comm_ops: np.ndarray,
        send_ops_forward: np.ndarray,
        send_ops_backward: np.ndarray,
    ) -> StressTestData:
        # TODO check input data
        # [rank, step, value]
        ret = StressTestData(
            local_forward=local_forward,
            local_backward=local_backward,
            comm_ops=comm_ops,
            send_ops_forward=send_ops_forward,
            send_ops_backward=send_ops_backward,
            gather_local_forward=None,
            gather_local_backward=None,
            gather_comm_ops=None,
            gather_send_forward=None,
            gather_send_backward=None,
        )
        if not self.is_pipeline_parallel or self.is_last_stage:
            ret.gather_local_forward = self.all_gather_func(
                ret.local_forward, group=self.dp_group
            )
        ret.gather_local_backward = self.all_gather_func(
            ret.local_backward, group=self.dp_group
        )

        ret.gather_comm_ops = self.all_gather_func(ret.comm_ops, group=self.dp_group)
        if self.is_pipeline_parallel:
            ret.gather_send_forward = self.all_gather_func(
                ret.send_ops_forward, group=self.dp_group
            )
            ret.gather_send_ops_b = self.all_gather_func(
                ret.send_ops_backward, group=self.dp_group
            )
        return ret

    def analyze_dp_failures(
        self, data: StressTestData, result: StressTestResult
    ) -> StressTestResult:
        def _merge_compare_ranks(*result):
            # return [step, [ranks]]
            compare_ranks = []
            for res in result:
                # res : [step, [ranks]]
                if res is None:
                    continue
                if not compare_ranks:
                    compare_ranks = [[] for _ in range(len(res))]
                for step, ranks in enumerate(res):
                    compare_ranks[step].extend(ranks)
            ret = list(map(lambda x: list(set(x)), compare_ranks))
            return ret

        def _merge_status(*result):
            # return [step, status]
            status = []
            for res in result:
                # res : [step, status]
                if res is None:
                    continue
                if not status:
                    status = [[]] * len(res)
                for step, s in enumerate(res):
                    status[step].append(s.value)
            ret = list(map(lambda x: StressTestDPCompareStatus(max(x)), status))
            return ret

        compare_ranks_forward = None
        if not self.is_pipeline_parallel or self.is_last_stage:
            loss_diff = self._compare_local_computation(
                data.local_forward,
                data.gather_local_forward,
                self.cur_rank,
                self.dp_domain_ranks,
            )
            logger.info("StressTest - evaluating loss difference.")
            (
                result.dp_compare_forward_match,
                result.dp_compare_backward_match,
                result.dp_status_forward,
                compare_ranks_forward,
            ) = self._evaluate_local_comparision(loss_diff, self.dp_domain_ranks)
        device_norm_diff = self._compare_local_computation(
            data.local_backward,
            data.gather_local_backward,
            self.cur_rank,
            self.dp_domain_ranks,
        )
        logger.info("StressTest - evaluating device norm difference.")
        (
            result.dp_compare_forward_match,
            result.dp_compare_backward_match,
            result.dp_status_backward,
            compare_ranks_backward,
        ) = self._evaluate_local_comparision(device_norm_diff, self.dp_domain_ranks)

        result.dp_status = _merge_status(
            result.dp_status_forward, result.dp_status_backward
        )
        compare_ranks = _merge_compare_ranks(
            compare_ranks_forward, compare_ranks_backward
        )
        result.compare_rank_idx = [ranks[0] if ranks else -1 for ranks in compare_ranks]

        return result

    def get_pp_compare_rank(self, result: StressTestResult):
        local_status = np.array([s.value for s in result.dp_status])
        gather_local_status = self.all_gather_func(local_status, group=self.pp_group)
        # [step, rank]
        gather_local_status = gather_local_status.transpose()
        pp_status = gather_local_status.max(axis=1)
        result.pp_status = [StressTestDPCompareStatus(s) for s in pp_status.tolist()]

        gather_pp_status = self.all_gather_func(
            np.array([s.value for s in result.pp_status]), group=self.dp_group
        )
        gather_pp_status = gather_pp_status.transpose()
        compare_ranks = np.argmax(gather_pp_status, axis=1)
        compare_ranks[
            gather_local_status[compare_ranks]
            < StressTestDPCompareStatus.MAJORITY_AGREE.value
        ] = -1
        result.compare_rank_idx = compare_ranks.tolist()
        return result

    def analyze_stage_failures(
        self, data: StressTestData, result: StressTestResult
    ) -> StressTestResult:
        cur_rank_idx = self.dp_domain_ranks.index(self.cur_rank)

        failed_stage_forward = np.array(
            list(map(lambda x: self.cur_stage if x else -1, result.pp_status))
        )
        min_rank_idx_forward, _ = self._find_first_differing_rank(
            data.gather_send_forward,
            cur_rank_idx,
            np.array(result.compare_rank_idx),
            self.pp_group,
            flip=False,
        )
        failed_stage_forward[min_rank_idx_forward == cur_rank_idx] = self.cur_stage

        failed_stage_backward = np.array(
            list(
                map(
                    lambda x: 0
                    if x.value > StressTestDPCompareStatus.MAJORITY_AGREE.value
                    and self.cur_stage == 0
                    else -1,
                    result.dp_status_backward,
                )
            )
        )
        min_rank_idx_backward, _ = self._find_first_differing_rank(
            data.gather_send_backward,
            cur_rank_idx,
            np.array(result.compare_rank_idx),
            self.pp_group,
            flip=True,
        )
        failed_stage_backward[min_rank_idx_backward == cur_rank_idx] = self.cur_stage

        gather_failed_stage = self.all_gather_func(
            np.array([failed_stage_forward, failed_stage_backward])
        )  # [rank, 2, step]

        gather_failed_stage = np.transpose(gather_failed_stage, (2, 0, 1))
        gather_failed_stage[:, :, 0] = np.where(
            gather_failed_stage[:, :, 0] < 0, np.inf, gather_failed_stage[:, :, 0]
        )

        failed_stage_forward = np.min(gather_failed_stage[:, :, 0], axis=1)
        failed_stage_backward = np.max(gather_failed_stage[:, :, 1], axis=1)
        failed_stage = np.array([-1] * len(failed_stage_forward))
        failed_stage[failed_stage_backward != -1] = failed_stage_backward[
            failed_stage_backward != -1
        ]
        failed_stage[failed_stage_forward != np.inf] = failed_stage_forward[
            failed_stage_forward != np.inf
        ]

        result.failed_stage = failed_stage.tolist()
        return result

    def analyze_tp_failures(self, data: StressTestData, result: StressTestResult):
        cur_rank_idx = self.dp_domain_ranks.index(self.cur_rank)

        if result.failed_stage:
            compare_rank_idx = list(
                map(
                    lambda idx, stage: idx if stage == self.cur_stage else -1,
                    result.compare_rank_idx,
                    result.failed_stage,
                )
            )
        else:
            compare_rank_idx = result.compare_rank_idx
        if all(idx == -1 for idx in compare_rank_idx):
            return result
        min_rank_idx, count = self._find_first_differing_rank(
            data.gather_comm_ops,
            cur_rank_idx,
            np.array(compare_rank_idx),
            self.stage_group,
        )
        result.failed_rank_idx = min_rank_idx.tolist()
        return result

    def _find_first_differing_rank(
        self, gather_ops, cur_rank_idx, cmp_rank_idx, group, flip=False
    ):
        gather_ops = gather_ops.transpose(1, 0, 2)  # [step, rank, value]
        min_diff = np.full((gather_ops.shape[0],), -1)
        # 忽略无法比较的rank，使其通过监测
        cmp_rank_idx[cmp_rank_idx < 0] = cur_rank_idx
        # cur_rank_idx: int, cmp_rank_idx: np.ndarray
        ops_diff = (
            gather_ops[:, cur_rank_idx, :]
            == gather_ops[np.arange(gather_ops.shape[0]), cmp_rank_idx, :]
        )  # [step, value]
        mask = ~ops_diff.all(axis=1)
        min_diff[mask] = np.argmax(~ops_diff[mask], axis=1)

        gather_min_diff = self.all_gather_func(min_diff, group)  # [rank, step]
        gather_min_diff = gather_min_diff.transpose(1, 0)
        gather_min_diff[gather_min_diff < 0] = np.inf
        gather_min_diff = np.flip(gather_min_diff, axis=1) if flip else gather_min_diff

        min_rank_idx = np.argmin(gather_min_diff, axis=1)  # [step,]
        mask = (
            gather_min_diff[np.arange(gather_min_diff.shape[0]), min_rank_idx] == np.inf
        )  # [step,]
        min_rank_idx[mask] = -1
        # expect only one min idx
        min_count = np.sum(gather_min_diff == min_rank_idx, axis=1)

        return min_rank_idx, min_count

    def _compare_local_computation(
        self, local_data, gather_data, cur_rank, dp_domain_ranks
    ):
        # ret = defaultdict(lambda: {"mismatch": [], "match": []})

        eq = local_data == gather_data
        if eq.ndim > 2:
            eq = eq.all(axis=tuple(range(2, eq.ndim)))
        # rank
        ret = [{"mismatch": [], "match": []} for _ in range(eq.shape[1])]
        for i in range(eq.shape[0]):
            if cur_rank == dp_domain_ranks[i]:
                continue
            # step
            for j, e in enumerate(eq[i]):
                if not e:
                    ret[j]["mismatch"].append(
                        {
                            "local_data": local_data[j],
                            "redundant_data": gather_data[i][j],
                            "redundant_rank": dp_domain_ranks[i],
                        }
                    )
                else:
                    ret[j]["match"].append(
                        {
                            "local_data": local_data[j],
                            "redundant_data": gather_data[i][j],
                            "redundant_rank": dp_domain_ranks[i],
                        }
                    )
        return ret

    def _evaluate_local_comparision(self, diff_data, dp_domain_ranks):
        def _get_compare_ranks(data):
            data_rank_map = defaultdict(list)

            for item in data["mismatch"]:
                k = item["redundant_data"]
                if isinstance(k, np.ndarray):
                    k = k.tobytes()
                data_rank_map[k].append(item["redundant_rank"])
            for item in data["match"]:
                k = item["redundant_data"]
                if isinstance(k, np.ndarray):
                    k = k.tobytes()
                data_rank_map[k].append(item["redundant_rank"])
            ranks_group = sorted(data_rank_map.values(), key=lambda x: len(x))
            if not ranks_group:
                return []
            if len(ranks_group) == 1:
                return ranks_group[0]
            else:
                if (
                    len(ranks_group[-1]) > len(ranks_group[-2])
                    and len(ranks_group[-1]) > 1
                ):
                    return ranks_group[-1]
                else:
                    return []

        match = [{} for _ in range(len(diff_data))]
        mismatch = [{} for _ in range(len(diff_data))]
        status = [StressTestDPCompareStatus.UNKNOWN] * len(diff_data)
        compare_ranks = [[] for _ in range(len(diff_data))]
        for step, data in enumerate(diff_data):
            mismatch_num = len(data["mismatch"])
            if mismatch_num == 0:
                match[step] = {
                    item["redundant_rank"]: item["redundant_data"]
                    for item in data["match"]
                }
                status[step] = StressTestDPCompareStatus.ALL_PASS
            elif mismatch_num == 1:
                match[step] = {
                    item["redundant_rank"]: item["redundant_data"]
                    for item in data["match"]
                }
                mismatch[step] = {
                    item["redundant_rank"]: item["redundant_data"]
                    for item in data["mismatch"]
                }
                status[step] = StressTestDPCompareStatus.MAJORITY_AGREE
                if len(dp_domain_ranks) <= 2:
                    status[step] = StressTestDPCompareStatus.CANNOT_DETERMINE
            else:
                compare_ranks[step] = _get_compare_ranks(data)
                if not compare_ranks[step]:
                    status[step] = StressTestDPCompareStatus.CANNOT_DETERMINE
                    continue
                mismatch[step] = {
                    item["redundant_rank"]: item["redundant_data"]
                    for item in data["mismatch"]
                }
                match[step] = {
                    item["redundant_rank"]: item["redundant_data"]
                    for item in data["match"]
                }
                status[step] = (
                    StressTestDPCompareStatus.MAJORITY_DISAGREE
                    if compare_ranks[step][0] in mismatch[step]
                    else StressTestDPCompareStatus.MAJORITY_AGREE
                )
        return match, mismatch, status, compare_ranks
