# ARC Prize 2026 — ARC-AGI-3 local dev workflow.
#
# Five commands cover the whole loop:
#   make setup        # one-time: venv + arc-agi + clone framework
#   make play-local   # fast inner loop: run agent/my_agent.py on a real game
#   make pull-sample  # fetch the official Stochastic Goose sample for reference
#   make submit       # build notebook from agent/my_agent.py + push to Kaggle
#   make status       # tail the latest Kaggle run

PYTHON          ?= python3.12
VENV            := .venv
VENV_PY         := $(VENV)/bin/python
VENV_PIP        := $(VENV)/bin/pip
# Read the project-local token at recipe time and expose it as KAGGLE_API_TOKEN
# (the only env var the modern Kaggle CLI honours for token auth).
KAGGLE          := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle
FRAMEWORK_REPO  := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR   := vendor/ARC-AGI-3-Agents
COMP_SLUG       := arc-prize-2026-arc-agi-3
GAME            ?=
SERVE_PORT      ?= 8001
SITE_PORT       ?= 8080
EVAL_PORT       ?= 8090
STEPS           ?= 200

.PHONY: help setup pbg pbg-replay pbg-test pbg-lint pbg-import-logs pbg-human-log pbg-metrics pbg-postmortem arcnav rulebook rulebook-bench rulebook-notebook rulebook-submit play-local pull-sample notebook submit status kaggle-log wheels llm-venv smoke-local verify-local serve exp-new exp-run exp-summary bench dashboard site site-publish eval-site eval-site-install eval-site-uninstall clean _check-kaggle

_check-kaggle:
	@if [ ! -s .kaggle/access_token ]; then \
	    echo "ERROR: .kaggle/access_token is missing or empty."; \
	    echo "       Generate a token at https://www.kaggle.com/settings (API → Create New Token)"; \
	    echo "       and save it as a one-line file at: $(PWD)/.kaggle/access_token"; \
	    exit 1; \
	fi

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Vars: PYTHON=$(PYTHON)  GAME=$(GAME)  STEPS=$(STEPS)"

setup: ## One-time install: venv, arc-agi, kaggle CLI, clone framework
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "arc-agi>=0.9.6" "kaggle>=2.2" python-dotenv pandas pyarrow
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then \
	    mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); \
	else \
	    git -C $(FRAMEWORK_DIR) pull --ff-only; \
	fi
	@# Slim agents/__init__.py so we don't need langgraph/langsmith/smolagents/etc.
	@# (Same trick the official Stochastic Goose sample uses on Kaggle.)
	@$(VENV_PY) scripts/slim_framework.py
	@echo ""
	@echo "Setup complete. Try:  make play-local"

rulebook: ## Rulebook agent (hypothesis rulebook + deterministic predictor + model choice): make rulebook [GAME=ls20,tn36] [MINUTES=12] [JOBS=2] [TAG=x] [NOMODEL=1]
	$(VENV_PY) scripts/run_rulebook.py --games $(or $(GAME),ls20) --minutes $(or $(MINUTES),12) --jobs $(or $(JOBS),2) --tag "$(TAG)" $(if $(NOMODEL),--no-model,) $(if $(MODE),--mode $(MODE),)

rulebook-bench: ## Long-budget measurement: REPEATS passes over GAME (default all 25) at MINUTES/game (default 60), JOBS concurrent, MODE coder: make rulebook-bench [REPEATS=3]
	for i in $$(seq 1 $(or $(REPEATS),3)); do \
	    $(VENV_PY) scripts/run_rulebook.py --games $(or $(GAME),all) --minutes $(or $(MINUTES),60) --jobs $(or $(JOBS),6) --level-actions $(or $(LEVEL_ACTIONS),400) --mode $(or $(MODE),coder) --tag "bench-$(or $(MODE),coder)-$(or $(MINUTES),60)m-pass$$i"; \
	done
	$(VENV_PY) scripts/rulebook_summary.py --tag bench-

rulebook-notebook: ## Build notebooks/rulebook/rulebook_submission.ipynb (rulebook agent + arcnav library + in-notebook vLLM)
	$(VENV_PY) scripts/build_rulebook_notebook.py

rulebook-submit: rulebook-notebook _check-kaggle ## Build and push the rulebook kernel (commit = 2-game smoke; leaderboard submit is manual on the web)
	$(KAGGLE) kernels push -p notebooks/rulebook/

pbg: ## pbg system (perception -> probe -> world-model lab -> goal -> planner, docs/027): make pbg [GAME=ls20,tn36] [MINUTES=10] [JOBS=2] [TAG=x] [NOLLM=1] [BUDGET=2000] [FRESH=1] [LLM=opus] [POLICY=hypothesis] [EXPLORE=20]
	$(if $(LLM),PBG_LLM_CONFIG=pbg/llm/llm-$(LLM).yaml,) $(VENV_PY) -m pbg.harness.online_runner --games $(or $(GAME),ls20) --minutes $(or $(MINUTES),10) --jobs $(or $(JOBS),2) --tag "$(TAG)" $(if $(NOLLM),--no-llm,) $(if $(BUDGET),--budget $(BUDGET),) $(if $(FRESH),--fresh,) $(if $(POLICY),--policy $(POLICY),) $(if $(EXPLORE),--explore $(EXPLORE),)

pbg-replay: ## Replay harness over recorded logs (perception/semantics/induction/goals/plans, no live env): make pbg-replay [LOG=pbg/data/human_logs/agent/ls20/raw.jsonl] [LLM=1]
	$(VENV_PY) -m pbg.harness.replay_runner $(or $(LOG),pbg/data/human_logs/agent/ls20/raw.jsonl) $(if $(LLM),--llm,)

pbg-test: pbg-lint ## Unit tests (+ integration tests when environment_files/ exists)
	$(VENV_PY) -m pytest pbg/tests -q

pbg-lint: ## no-game-id-branch lint (spec §13): fails when any module branches on a game id string
	$(VENV_PY) pbg/tools/lint_no_game_id.py

pbg-import-logs: ## Re-execute a recorded rulebook run into pbg/data/human_logs/agent/<game>/raw.jsonl: make pbg-import-logs RUN=<run id> [GAME=ls20,tn36]
	$(VENV_PY) pbg/tools/import_run_logs.py --run $(RUN) $(if $(GAME),--games $(GAME),)

pbg-human-log: ## Record a human play log in the terminal: make pbg-human-log GAME=ls20 [PLAYER=me]
	$(VENV_PY) pbg/tools/collect_human_log.py $(or $(GAME),ls20) --player $(or $(PLAYER),human)

pbg-postmortem: ## Per-level post-mortem of one game in a run: make pbg-postmortem RUN=<run id> GAME=ls20
	$(VENV_PY) -m pbg.harness.postmortem $(RUN) $(GAME)

pbg-metrics: ## Metrics + bottleneck attribution of a pbg run JSON: make pbg-metrics RUN=experiments/pbg/results/run-....json
	$(VENV_PY) -m pbg.harness.metrics $(RUN)


arcnav: ## Play games with our arcnav agent against the local vLLM server: make arcnav [GAME=ls20,vc33] [MINUTES=20] [JOBS=2] [TAG=x]
	$(VENV_PY) scripts/run_arcnav.py --games $(or $(GAME),ls20) --minutes $(or $(MINUTES),20) --jobs $(or $(JOBS),2) --tag "$(TAG)"

play-local: ## Run agent/my_agent.py against ALL games (or GAME=ls20 for a single one)
	$(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify-local: ## Quick smoke test: 50 steps on ls20 + vc33 only
	$(VENV_PY) scripts/play_local.py --game ls20,vc33 --max-steps 50

list-games: ## Show all available games
	$(VENV_PY) scripts/play_local.py --list

pull-sample: _check-kaggle ## Download the official Stochastic Goose sample notebook for reference
	mkdir -p reference/stochastic-goose
	$(KAGGLE) kernels pull inversion/arc3-sample-submission-stochastic-goose \
	    -p reference/stochastic-goose -m
	@echo "Open reference/stochastic-goose/*.ipynb for the canonical pattern."

notebook: ## Splice agent/my_agent.py into notebooks/submission.ipynb
	$(VENV_PY) scripts/build_notebook.py

submit: notebook _check-kaggle ## Build notebook and push to Kaggle (one-line submission)
	@grep -q REPLACE_WITH_YOUR_USERNAME notebooks/kernel-metadata.json && { \
	    echo "ERROR: edit notebooks/kernel-metadata.json and replace REPLACE_WITH_YOUR_USERNAME"; \
	    exit 1; } || true
	$(KAGGLE) kernels push -p notebooks/
	@echo ""
	@echo "Pushed. Track it with:  make status"

status: _check-kaggle ## Show the status of your most recent Kaggle kernel run
	@KERNEL_ID=$$(python3 -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

kaggle-log: _check-kaggle ## Print the Kaggle kernel log (KERNEL=notebooks|notebooks/wheels|owner/slug, GREP=regex)
	$(VENV_PY) scripts/kaggle_log.py $(or $(KERNEL),notebooks) $(if $(GREP),--grep "$(GREP)") --tail $(or $(TAIL),60)

wheels: _check-kaggle ## Push the vLLM wheel-cache kernel (internet on; output attached to the submission)
	$(KAGGLE) kernels push -p notebooks/wheels/

llm-venv: ## One-time: separate venv with vLLM for local in-process smoke tests (.venv-llm, ~5GB)
	$(PYTHON) -m venv .venv-llm
	.venv-llm/bin/pip install --upgrade pip
	.venv-llm/bin/pip install vllm "arc-agi>=0.9.6" python-dotenv

smoke-local: ## Local GPU smoke test of the planner backend: make smoke-local MODEL_PATH=~/models/x [BACKEND=vllm|hf] [AGENT=path]
	@test -n "$(MODEL_PATH)" || { echo "usage: make smoke-local MODEL_PATH=<hf checkpoint dir> [BACKEND=vllm|hf]"; exit 1; }
	.venv-llm/bin/python scripts/smoke_llm.py --model-path $(MODEL_PATH) $(if $(BACKEND),--backend $(BACKEND)) $(if $(AGENT),--agent $(AGENT))

serve: ## Host the ARC-AGI-3 API locally on http://localhost:8001 (framework default)
	$(VENV_PY) scripts/serve_local.py --port $(SERVE_PORT)


exp-new: ## Scaffold a new experiment: make exp-new NAME=greedy-search [FROM=v001]
	$(VENV_PY) scripts/new_experiment.py $(NAME) $(if $(FROM),--from $(FROM))

exp-run: ## Run an experiment and record results: make exp-run NAME=v001 [GAME=ls20] [STEPS=200] [JOBS=8] [SEED=7] [TAG=x]
	$(VENV_PY) scripts/run_experiment.py $(NAME) $(if $(GAME),--game $(GAME)) $(if $(STEPS),--max-steps $(STEPS)) $(if $(JOBS),--jobs $(JOBS)) $(if $(SEED),--seed $(SEED)) $(if $(TAG),--tag $(TAG))

exp-summary: ## Compare all experiments (table + experiments/summary.json)
	$(VENV_PY) scripts/exp_summary.py


bench: ## Benchmark version(s) over seeds + publish site: make bench NAME=v006 [GAME=..] [STEPS=3000] [SEEDS=1337,7,42] [JOBS=8] (ALL=1: every version)
	$(VENV_PY) scripts/benchmark.py $(or $(NAME),$(ONLY)) $(if $(ALL),--all) $(if $(GAME),--game $(GAME)) $(if $(STEPS),--max-steps $(STEPS)) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(JOBS),--jobs $(JOBS))

dashboard: ## Rebuild the benchmark site (experiments/site/: index + page per benchmark)
	$(VENV_PY) scripts/exp_summary.py
	$(VENV_PY) scripts/build_dashboard.py
	@echo "Open: file://$(PWD)/experiments/site/index.html"


site: ## Serve the benchmark site at http://localhost:8080, auto-rebuilding on refresh (SITE_PORT=)
	$(VENV_PY) scripts/serve_site.py --port $(SITE_PORT)


eval-site: ## Replay: local viewer of rulebook/pbg runs incl. ones in progress (game -> level -> step boards) at http://localhost:8090 (EVAL_PORT=)
	$(VENV_PY) scripts/serve_eval.py --port $(EVAL_PORT)

EVAL_UNIT       := arc-eval-site.service
EVAL_UNIT_DIR   := $(HOME)/.config/systemd/user

eval-site-install: ## Install + enable the eval viewer as a systemd user service (starts at boot, EVAL_PORT=)
	mkdir -p $(EVAL_UNIT_DIR)
	sed -e 's|__ROOT__|$(CURDIR)|g' -e 's|__PORT__|$(EVAL_PORT)|g' scripts/systemd/$(EVAL_UNIT) > $(EVAL_UNIT_DIR)/$(EVAL_UNIT)
	systemctl --user daemon-reload
	systemctl --user enable --now $(EVAL_UNIT)
	@loginctl show-user $(USER) -p Linger | grep -q 'Linger=yes' || echo "NOTE: run 'loginctl enable-linger $(USER)' so the service starts at boot without a login"
	systemctl --user --no-pager status $(EVAL_UNIT) | head -5

eval-site-uninstall: ## Stop, disable and remove the eval viewer systemd user service
	-systemctl --user disable --now $(EVAL_UNIT)
	rm -f $(EVAL_UNIT_DIR)/$(EVAL_UNIT)
	systemctl --user daemon-reload


site-publish: ## Publish the benchmark site + static Replay (/replay/) to GitHub Pages (gh-pages branch) [REPLAY=0] [REPLAY_ARGS="--since 20260928 --no-logs"]
	bash scripts/publish_site.sh


clean: ## Remove generated artefacts (venv, downloaded games, vendored repos)
	rm -rf $(VENV) vendor environment_files recordings notebooks/submission.ipynb \
	       reference logs.log __pycache__ .pytest_cache
