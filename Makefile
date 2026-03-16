# Research Workflow Standard
.PHONY: setup test benchmark clean

setup:
	pip install -r requirements.txt

test:
	pytest tests/

benchmark:
	python benchmarks/profile_scaling.py --max_length 65536

clean:
	rm -rf __pycache__ .pytest_cache outputs/
