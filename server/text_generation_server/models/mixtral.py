import torch
import torch.distributed

from opentelemetry import trace
from transformers.models.llama import LlamaTokenizer, LlamaTokenizerFast
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.modeling_moe_mistral import (
    MixtralForCausalLM,
)
from text_generation_server.models.custom_modeling.configuration_moe_mistral import MixtralConfig
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from transformers.configuration_utils import PretrainedConfig

tracer = trace.get_tracer(__name__)


class FlashMixtral(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashMistral is only available on GPU")

        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            tokenizer = LlamaTokenizerFast.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )

        config = MixtralConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        print("Loading weights")
        # filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        # weights = Weights(filenames, device, dtype, process_group=self.process_group)
        # if config.quantize == "gptq":
        #     weights._set_gptq_params(model_id)

        model = MixtralForCausalLM.from_pretrained(
            model_id, config=config, low_cpu_mem_usage=True,
            device_map="auto", trust_remote_code=True)
        # model = MixtralForCausalLM(config, weights)  # , weights
        print(dir(model))
        print(dir(model.model))

        torch.distributed.barrier(group=self.process_group)
        super(FlashMixtral, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=8,
            head_size=32,
            # num_kv_heads=model.model.num_key_value_heads,
            # head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            window_size=(4096, 4096),
        )
