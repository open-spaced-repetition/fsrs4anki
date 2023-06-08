VERSION=${shell cat version.txt}
WHEEL_FILE=dist/fsrs4anki_optimizer-${VERSION}-py3-none-any.whl

${WHEEL_FILE}:
	python setup.py bdist_wheel

.PHONEY: install
install: ${WHEEL_FILE}
	pip install ${WHEEL_FILE}
	