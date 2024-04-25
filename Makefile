install:
	pip install -r requirements.txt
	pre-commit install

format_file:
	autopep8 --aggressive --aggressive --in-place ${file}

format_all_files:
	find . -name "*.py" -exec autopep8 --aggressive --aggressive --in-place {} \;

run_precommit:
	pre-commit run --all-files

huggingface_login:
	huggingface-cli login