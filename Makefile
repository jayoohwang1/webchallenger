.PHONY: prepare

prepare:
	@echo "Preparing evaluation environment..."
	bash scripts/prepare_eval.sh
	@echo "Evaluation environment is ready."
	
clean:
	@echo "Cleaning up evaluation environment..."
	bash scripts/clean_eval.sh