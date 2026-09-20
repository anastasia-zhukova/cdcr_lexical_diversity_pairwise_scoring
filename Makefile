# Sweeps of the pipeline steps over the experiment configs in config/ (config/_archive and the
# test_*.yaml smoke configs are ignored; pass CONFIGS=... to run those explicitly).
#
#   make gen-pairs                 step 1 for every config
#   make embed                     step 2 for every config
#   make train | inference | scoring
#   make all                       every step, each swept over every config before the next step starts
#   make pipeline CONFIG=single-random-cd2cr      all five steps for one config
#   make embed CONFIGS="single-random-cd2cr single-random-gvc"   restrict a sweep
#   make gen-pairs CONFIGS="$(make list-configs | grep ^single)" e.g. one family only
#
# A failing config does not stop the sweep: the remaining configs still run and the failures are
# listed at the end (the target then exits non-zero). Each run's output is kept in
# $(LOG_DIR)/<step>/<config>.log.
#
# Every run is started under a cgroup memory cap (MEMORY_MAX, no swap) through systemd-run so that a
# misbehaving step cannot take the machine down; set MEMORY_MAX= (empty) to run without the cap.

PYTHON      ?= .venv/bin/python
PACKAGE     := cdcr_lexical_diversity_pairwise_scoring
CONFIG_DIR  := config
LOG_DIR     ?= sweep_logs
MEMORY_MAX  ?= 12G
ALL_CONFIGS := $(sort $(notdir $(basename $(wildcard $(CONFIG_DIR)/*.yaml))))
CONFIGS     ?= $(filter-out test_%,$(ALL_CONFIGS))

# Pair sampling and the order of the CoNLL output go through Python sets: pinning the hash seed makes
# a sweep reproducible run to run. Override with PYTHONHASHSEED=random to get the interpreter default.
export PYTHONHASHSEED ?= 0
export MLFLOW_DISABLE_AGENT_HINT := 1

STEPS := gen-pairs embed train inference scoring

script_gen-pairs := preprocess_gen_pairs
script_embed     := preprocess_embed
script_train     := train
script_inference := inference_clustering
script_scoring   := scoring

ifneq ($(strip $(MEMORY_MAX)),)
RUNNER := systemd-run --user --scope -q -p MemoryMax=$(MEMORY_MAX) -p MemorySwapMax=0 --
else
RUNNER :=
endif

.PHONY: help all pipeline list-configs $(STEPS)

help:
	@sed -n '1,18p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'

list-configs:
	@printf '%s\n' $(CONFIGS)

# make <step>: run one script over every config, keep going on failure, summarise at the end
$(STEPS):
	@mkdir -p $(LOG_DIR)/$@
	@failed=""; \
	for config in $(CONFIGS); do \
		printf '==> %-10s %s\n' "$@" "$$config"; \
		if ! $(RUNNER) $(PYTHON) $(PACKAGE)/$(script_$@).py --config-name "$$config" > "$(LOG_DIR)/$@/$$config.log" 2>&1; then \
			printf '    FAILED (see %s)\n' "$(LOG_DIR)/$@/$$config.log"; \
			failed="$$failed $$config"; \
		fi; \
	done; \
	if [ -n "$$failed" ]; then printf '\n%s failed for:%s\n' "$@" "$$failed"; exit 1; fi; \
	printf '\n%s done for %s configs\n' "$@" "$(words $(CONFIGS))"

# make all: sweep the whole pipeline step by step; a step's sweep must fully succeed before the next
all:
	@for step in $(STEPS); do $(MAKE) --no-print-directory $$step || exit 1; done

# make pipeline CONFIG=<name>: the five steps in order for a single config
pipeline:
	@test -n "$(CONFIG)" || { echo "usage: make pipeline CONFIG=<config name>"; exit 2; }
	@for step in $(STEPS); do $(MAKE) --no-print-directory $$step CONFIGS="$(CONFIG)" || exit 1; done
