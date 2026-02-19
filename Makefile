.PHONY: setup run clean logs

setup:
	@echo "Syncing dependencies with uv..."
	uv sync
	@echo "Ensuring cache directory exists..."
	mkdir -p ./hf_cache
	@if [ ! -f .env ]; then cp .env.example .env; echo "\nWARNING: Created .env file. Open it and add your HF_TOKEN!"; fi

run:
	@echo "Starting vLLM server..."
	uv run --env-file .env vllm serve meta-llama/Meta-Llama-3-8B-Instruct --dtype auto

clean:
	@echo "Wiping Hugging Face cache to free up your 30GB EBS drive..."
	rm -rf ./hf_cache/*

logs:
	@echo "Tailing AWS startup logs..."
	sudo tail -f /var/log/cloud-init-output.log
