### Motivation of isFit classifiers
Separate FPGA compilation using Partial Reconfiguration needs to assign synthesized netlists
to appropriate reconfigurable partitions to compile in parallel.
This assignment includes a subproblem to determine whether a netlist could be successfully mapped
on a speciific reconfigurable partition or not.
A simple approach is to use a capacity-based hard constraint.
For example, if the post-synthesis resource estimates are less than 70% of the resources availalbe
in the reconfigurable partition, we assume that the netlist can be safely mapped.
However, this approach may not work well based on reasons below:
- Irregular columnar resource distribution on modern FPGAs
- AMD-Xilinx PR technology allows static routing to route over reconfigurable regions
- Every design (synthesized netlist) has different routing complexity


### Overview
There are total 36 reconfigurable regions including hierarchical regions (with nested DFX). 
We accept that each reconfigurable region is different and generate
classifier *per* each reconfigurable region (total 36 classifiers).
A range of different reconfigurable modules are tried on each reconfigurable region
to generate the training/test data. 
We 
selected netlist with over 60% of the reconfigurable region in LUT utilization,
ran implementations,
and recorded the results in ./rpt_dir/impl_results/{FREQUENCY}/csv/ dir.
If all LUT, BRAM and DSP post-synthesis resource estimates are
lower than 60% of the available resources of the reconfigurable regions, 
we will assume that the netlist can be successfully mapped on a specific reconfigurable partition.

Features include post-synthesis resource estimates (LUT, BRAM and DSP) from `report_utilization` and
complexity characterstics (Rent, average fanout, and total instances) from `report_design_analysis` command.

When generating implementation results, we didn't use any *directive* for `place_design` and `route_design` command.
For this reason, in isFit.py, we do not consider "Timing violation" as implementation failure,
hoping that other directives can significantly improve the timing.
In our case, False Positives (the classifier predicts that the netlist would fail in implementation, but it succeeds)
are relatively acceptable.
However, False Negatives (our classifier predicts that the netlist would succeed in implementation, but it fails)
are not acceptable as they could require recompilation of the page, slowing down the parallel compilation strategy.
Therefore, we adjust our classifiers to at least match a target value of
*recall*, (True Positives/(True Positives+False Negatives)) and
evaluate whether the classifiers still perform better than hard constraints in
*recall*, (True Positives/(True Positives+False Positives)).

### How to run
```
python isFit.py -f 200 -t 0.96
```
The command will generate classifiers in .pickle file format in ./rpt_dir/classifier/200MHz/96/ dir
when 200 is the frequency the module will run and 0.96 is the recall threshold.

### Limitations and Future work
We support 5 different clock frequencies (200MHz, 250MHz, 300MHz, 350MHz, 400MHz) and
has total 36 PR regions (single + double + quad). Thus, the total number of
trained classifiers is 180.
The average number of training/test datasets for each region is about 2,500.
The generation of synthesized netlists take
more than 20,000 synthesis runs, and the generation of
place/route results take about 450,000 experiments run in the compute server.
Assuming we have a large compute server that can run 45 Vivado compile runs,
each compute node is in charge or 450,000/45 = 10,000 Vivado compiles.
Assuming each takes about 10 minutes, it should take 100,000 minutes ~= 69 days.

If 1) we use FPGA devices with more homogeneous resource distribution like AMD-Xilinx Veral or Intel Agilex series
and 2) we can entire remove static routing over reconfigurable regions,
we can train one classifier per size.
Then, in our system, we need only 3(single, double,quad) * 5(frequencies) = 15 classifiers.
With the same compute resources, it would take only 5.75 days to generate classifiers.

