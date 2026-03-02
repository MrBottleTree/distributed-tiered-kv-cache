.PHONY: setup run clean logs

setup:
	@echo "Syncing dependencies with uv..."
	uv sync
	@echo "Ensuring cache directory exists..."
	mkdir -p ./hf_cache
	@if [ ! -f .env ]; then cp .env.example .env; echo "\nWARNING: Created .env file. Open it and add your HF_TOKEN!"; fi

run:
	@echo "Starting vLLM server..."
	uv run --env-file .env vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
	  --dtype auto \
	  --max-model-len 8192 \
	  --gpu-memory-utilization 0.90 \
	  --enforce-eager

clean:
	@echo "Wiping Hugging Face cache to free up your 30GB EBS drive..."
	rm -rf ./hf_cache/*

logs:
	@echo "Tailing AWS startup logs..."
	sudo tail -f /var/log/cloud-init-output.log

run_quantized:
	@echo "Starting Official Meta Llama 3.1 8B on G5..."
	uv run --env-file .env vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
		--dtype bfloat16 \
		--max-model-len 20000 \
		--gpu-memory-utilization 0.90 \
		--enforce-eager
