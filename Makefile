INDIGO_ROOT ?= indigo
GRAPH_DIR   ?= indigo/large_inputs
GRAPH_PATH  := https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/
G1          := $(GRAPH_DIR)/rmat22.sym.egr
G2          := $(GRAPH_DIR)/r4-2e23.sym.egr
G3          := $(GRAPH_DIR)/in-2004.egr
G4          := $(GRAPH_DIR)/USA-road-d.USA.egr
G5          := $(GRAPH_DIR)/as-skitter.egr
G6          := $(GRAPH_DIR)/citationCiteseer.egr
TBL1_DB     := results/hirace_correctness_results.sqlite3
FIG5_DB     := results/hirace_perf_results_speedup.sqlite3
FIG6_DB     := results/hirace_perf_results.sqlite3
FIG6_DIR    := fig6-tests/cuda

.PRECIOUS: $(TBL1_DB) $(FIG5_DB) $(FIG6_DB)
.PHONY: all table1 fig5 fig6
all: table1 fig5 fig6
	
table1: | $(TBL1_DB)
	python scripts/gen_table1.py $(TBL1_DB)
	
$(TBL1_DB): # About 3-6 hours
	python scripts/hirace_experiments.py . $(INDIGO_ROOT)/indigo_sources $(INDIGO_ROOT)/input $(notdir $(TBL1_DB))

fig5: | $(FIG5_DB)
	python scripts/gen_fig5.py $(FIG5_DB)
	
$(FIG5_DB): $(G1) $(G2) $(G3) $(G4) $(G5) $(G6) # About 6-10 hours
	python scripts/hirace_experiments.py . $(INDIGO_ROOT)/indigo_sources/conditional_edge_neighbor $(INDIGO_ROOT)/large_inputs $(notdir $(FIG5_DB))

fig6: | $(FIG6_DB)
	python scripts/gen_fig6.py $(FIG6_DB)

$(FIG6_DB): # About 1-4 hours
	$(MAKE) -C $(FIG6_DIR)/backprop
	python scripts/fig6_experiments.py $(FIG6_DIR)/backprop/run_backprop $(notdir $(FIG6_DB))
	$(MAKE) -C $(FIG6_DIR)/gaussian
	python scripts/fig6_experiments.py $(FIG6_DIR)/gaussian/run_gaussian $(notdir $(FIG6_DB))
	$(MAKE) -C $(FIG6_DIR)/srad/srad_v1
	python scripts/fig6_experiments.py $(FIG6_DIR)/srad_srad_v1/run_srad $(notdir $(FIG6_DB))
	$(MAKE) -C $(FIG6_DIR)/BlackScholes
	python scripts/fig6_experiments.py $(FIG6_DIR)/BlackScholes/run_BlackScholes $(notdir $(FIG6_DB))
	$(MAKE) -C $(FIG6_DIR)/fastWalshTransform
	python scripts/fig6_experiments.py $(FIG6_DIR)/fastWalshTransform/run_fastWalshTransform $(notdir $(FIG6_DB))
	$(MAKE) -C $(FIG6_DIR)/convolutionFFT2D
	python scripts/fig6_experiments.py $(FIG6_DIR)/convolutionFFT2D/run_convolutionFFT2D $(notdir $(FIG6_DB))


$(G1): | $(GRAPH_DIR)
	wget --no-check-certificate $(GRAPH_PATH)$(notdir $(G1)) -P $(GRAPH_DIR)

$(G2): | $(GRAPH_DIR)
	wget --no-check-certificate $(GRAPH_PATH)$(notdir $(G2)) -P $(GRAPH_DIR)

$(G3): | $(GRAPH_DIR)
	wget --no-check-certificate $(GRAPH_PATH)$(notdir $(G3)) -P $(GRAPH_DIR)

$(G4): | $(GRAPH_DIR)
	wget --no-check-certificate $(GRAPH_PATH)$(notdir $(G4)) -P $(GRAPH_DIR)

$(G5): | $(GRAPH_DIR)
	wget --no-check-certificate $(GRAPH_PATH)$(notdir $(G5)) -P $(GRAPH_DIR)

$(G6): | $(GRAPH_DIR)
	wget --no-check-certificate $(GRAPH_PATH)$(notdir $(G6)) -P $(GRAPH_DIR)


$(GRAPH_DIR):
	mkdir -p indigo/large_inputs
	
clean:
	$(RM) -r $(GRAPH_DIR)
	$(MAKE) -C $(FIG6_DIR)/backprop clean
	$(MAKE) -C $(FIG6_DIR)/gaussian clean
	$(MAKE) -C $(FIG6_DIR)/srad/srad_v1 clean
	$(MAKE) -C $(FIG6_DIR)/BlackScholes clean
	$(MAKE) -C $(FIG6_DIR)/fastWalshTransform clean
	$(MAKE) -C $(FIG6_DIR)/convolutionFFT2D clean

clean-all: clean
	$(RM) $(TBL1_DB)
	$(RM) $(FIG5_DB)
	$(RM) $(FIG6_DB)
	$(RM) results/compare*.png
