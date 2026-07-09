# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel as HFAutoModel

from nemo_automodel import NeMoAutoModelBiEncoder
from nemo_automodel._transformers.auto_model import _BaseNeMoAutoModelClass
from nemo_automodel._transformers.retrieval import BiEncoderModel
from nemo_automodel.components.loss.intermediate_distill import LayerCapture
from nemo_automodel.shared.utils import dtype_from_str


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer block list on a HuggingFace backbone."""
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers") and isinstance(model.model.layers, nn.ModuleList):
        return model.model.layers
    if (
        hasattr(model, "transformer")
        and hasattr(model.transformer, "h")
        and isinstance(model.transformer.h, nn.ModuleList)
    ):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers on model (tried .layers, .model.layers, .transformer.h)")


@contextmanager
def _default_torch_dtype(dtype: torch.dtype | None):
    if dtype is None:
        yield
        return
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _resolve_init_dtype(torch_dtype, config) -> torch.dtype | None:
    if torch_dtype == "auto":
        return dtype_from_str(getattr(config, "torch_dtype", None) or getattr(config, "dtype", None))
    return dtype_from_str(torch_dtype)


class _NeMoAutoModel(_BaseNeMoAutoModelClass, HFAutoModel):
    """Generic AutoModel variant with Automodel infrastructure support."""

    pass


_AUTOMODEL_RUNTIME_KWARGS = {
    "device_mesh",
    "moe_mesh",
    "distributed_config",
    "peft_config",
    "tp_plan",
    "compile_config",
    "pipeline_config",
    "qat_config",
    "moe_config",
    "activation_checkpointing",
    "fp8_config",
    "sdpa_method",
    "has_packed_sequence",
    "tp_size",
    "cp_size",
}


class StudentWithProjection(nn.Module):
    """Bi-encoder student with a trainable linear projection into teacher space."""

    def __init__(
        self,
        student: BiEncoderModel,
        teacher_hidden_size: int,
        capture_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.student = student
        student_hidden = int(self.student.model.config.hidden_size)
        self.student_hidden = student_hidden
        self.teacher_hidden = int(teacher_hidden_size)

        self.projection = nn.Linear(student_hidden, self.teacher_hidden, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection = self.projection.float()
        if torch.cuda.is_available():
            # Keep projection colocated with rank-local activations under distributed training.
            self.projection = self.projection.to(device=torch.cuda.current_device())

        self._capture = LayerCapture(detach=False)
        if capture_layers:
            self.attach_intermediate_capture(capture_layers)

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        teacher_hidden_size: int,
        pooling: str = "last",
        l2_normalize: bool = False,
        capture_layers: Sequence[int] | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "StudentWithProjection":
        student = NeMoAutoModelBiEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pooling=pooling,
            l2_normalize=l2_normalize,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(student=student, teacher_hidden_size=teacher_hidden_size, capture_layers=capture_layers)

    @classmethod
    def build_from_config(
        cls,
        pretrained_model_name_or_path: str,
        teacher_hidden_size: int,
        pooling: str = "last",
        l2_normalize: bool = False,
        capture_layers: Sequence[int] | None = None,
        trust_remote_code: bool = True,
        torch_dtype="auto",
        attn_implementation: str | None = None,
        random_init_seed: int = 42,
        **kwargs,
    ) -> "StudentWithProjection":
        """Build a student with the checkpoint architecture but freshly initialized weights."""
        config_kwargs = {
            key: kwargs.pop(key)
            for key in ("cache_dir", "revision", "token", "local_files_only")
            if key in kwargs
        }
        runtime_kwargs = {key: kwargs.pop(key) for key in _AUTOMODEL_RUNTIME_KWARGS if key in kwargs}
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )
        config.name_or_path = pretrained_model_name_or_path
        config._name_or_path = pretrained_model_name_or_path
        if pooling is not None:
            config.pooling = pooling
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation
        init_dtype = _resolve_init_dtype(torch_dtype, config)

        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices, enabled=True), _default_torch_dtype(init_dtype):
            torch.manual_seed(int(random_init_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(random_init_seed))
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
                **runtime_kwargs,
                **kwargs,
            }
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
            backbone = _NeMoAutoModel.from_config(config, **model_kwargs)

        student = BiEncoderModel(
            model=backbone,
            pooling=pooling,
            l2_normalize=l2_normalize,
        )
        return cls(student=student, teacher_hidden_size=teacher_hidden_size, capture_layers=capture_layers)

    def attach_intermediate_capture(self, layer_indices: Iterable[int]) -> "StudentWithProjection":
        self._capture.attach(_get_layers(self.student.model), layer_indices)
        return self

    def detach_intermediate_capture(self) -> None:
        self._capture.detach_hooks()

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        # Keep the save format HF-compatible for evaluator tooling: this stores
        # only the student backbone; the projection is saved by the recipe.
        self.student.save_pretrained(save_directory, **kwargs)

    def _encode(self, input_dict: dict) -> torch.Tensor:
        if not input_dict:
            return None
        embeds = self.student(input_dict)
        return embeds.contiguous()

    def forward(
        self,
        input_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        self._capture.reset()
        pooled = self._encode(input_dict)
        pooled_fp32 = pooled.float()

        with torch.amp.autocast(device_type=pooled.device.type, enabled=False):
            projected = self.projection(pooled_fp32)

        return pooled_fp32, projected, dict(self._capture.outputs)


class TeacherEmbeddingEncoder(nn.Module):
    """Frozen bi-encoder teacher with optional intermediate-layer capture."""

    def __init__(
        self,
        teacher: BiEncoderModel,
        capture_layers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.hidden_size = int(self.teacher.model.config.hidden_size)
        self._capture = LayerCapture(detach=True)
        if capture_layers:
            self.attach_intermediate_capture(capture_layers)

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str,
        pooling: str = "last",
        l2_normalize: bool = False,
        capture_layers: Sequence[int] | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "TeacherEmbeddingEncoder":
        teacher = NeMoAutoModelBiEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pooling=pooling,
            l2_normalize=l2_normalize,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(teacher=teacher, capture_layers=capture_layers)

    def attach_intermediate_capture(self, layer_indices: Iterable[int]) -> "TeacherEmbeddingEncoder":
        self._capture.attach(_get_layers(self.teacher.model), layer_indices)
        return self

    def detach_intermediate_capture(self) -> None:
        self._capture.detach_hooks()

    @torch.no_grad()
    def _encode(self, input_dict: dict) -> torch.Tensor:
        if not input_dict:
            return None
        embeds = self.teacher(input_dict)
        return embeds.contiguous()

    @torch.no_grad()
    def forward(self, input_dict: dict) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        self._capture.reset()
        pooled = self._encode(input_dict)
        return pooled.float(), dict(self._capture.outputs)
