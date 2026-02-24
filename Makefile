.PHONY: test run sweep pdf clean

# Run all tests
test:
	python -m pytest tests/ -v

# Run anchor config experiment (requires config.json)
run:
	python run_experiment.py --config config.json

# Run parameter sweep (placeholder — implemented in Phase 10)
sweep:
	@echo "Sweep not yet implemented (Phase 10)"

# Generate math verification PDF (placeholder — implemented in Phase 9)
pdf:
	@echo "PDF generation not yet implemented (Phase 9)"

# Clean generated artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf results/
