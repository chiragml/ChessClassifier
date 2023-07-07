ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SCRIPT_DIR:=$(ROOT_DIR)/scripts

lichess: lichess/lichess.onehot.model.pkl lichess/lichess.heuristics.model.pkl
alphazero: alphazero/both.onehot.model.pkl alphazero/both.heuristics.model.pkl

%.pkl: %.pgn
	$(SCRIPT_DIR)/pp.py pgn $< $@

%.pkl: %.csv
	$(SCRIPT_DIR)/pp.py csv $< $@

%.phi.pkl: %.pkl
	$(SCRIPT_DIR)/pp.py phi $< $@

%.heuristics.pkl: %.pkl
	$(SCRIPT_DIR)/pp.py heuristics $< $@

%.onehot.pkl: %.pkl
	$(SCRIPT_DIR)/pp.py onehot $< $@

%.model.pkl: %.pkl
	$(SCRIPT_DIR)/model.py $< $@
