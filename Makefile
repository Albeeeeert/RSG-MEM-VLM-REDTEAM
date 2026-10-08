.PHONY: install test demo inspect

install:
	python -m pip install -e .

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python -m rsg_mem demo --output runs/demo

inspect:
	PYTHONPATH=src python -m rsg_mem inspect --config configs/mock.yaml
