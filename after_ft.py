from peft import AutoPeftModelForCausalLM


path_to_adapter="Qwen-VL/output_qwen"
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="cuda",
    trust_remote_code=True
).eval()