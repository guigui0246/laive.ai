# laive.ai
Laive.Ai test repo

Requirements.txt
Most requirements also need user installed applications

## Deepseek abandon reason
Some versions of deepseek might return an error saying it cannot import `is_torch_greater_or_equal_than_1_13`
In this case go to `~/.cache/huggingface/modules/transformers_module/deepseek-ai/` and change `_1_13` to `_2_1` in every file and subfolder. It should appear twice.

## Other language abandon reason
I switched to API because either local only returned half a sentence or asked for 2TB or VRAM
