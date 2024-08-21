import torch
from transformers import GPTNeoForCausalLM


class GPTNeoWithIntermediates(GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config.output_hidden_states = True

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        hidden_states = outputs.hidden_states

        layer_outputs = []
        for i, hidden_state in enumerate(hidden_states):
            if i < len(self.transformer.h):
                mlp_hidden_output = self.transformer.h[i].mlp.c_fc(hidden_state)
                mlp_final_output = self.transformer.h[i].mlp.c_proj(mlp_hidden_output)

                layer_outputs.append({
                    "attn_out": hidden_state,
                    "mlp_hidden": mlp_hidden_output,
                    "mlp_final": mlp_final_output,
                    })


        return {"logits": outputs.logits, "layer_outputs": layer_outputs}

