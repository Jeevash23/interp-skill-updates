"""Write paper/generated/numbers.tex and the generated tables: every study number in the manuscript.

Each macro is computed from a saved, verified output: the two frozen studies under studies/ and the post hoc and
descriptive analyses under analysis/results/. `--check` regenerates everything in memory and fails if a committed
file differs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

from scipy.stats import t as student_t

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
P1 = PROJECT / "studies/1-feasibility-classification"
P1R = P1 / "04-Results"
P2 = PROJECT / "studies/2-controlled-histories"
P2A = P2 / "analysis"
AR = PROJECT / "analysis/results"
ARCHS = ("linear", "leaky")
NAMES = {"linear": "Linear", "leaky": "Leaky"}
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def read(p):
    return json.loads(Path(p).read_text())


def f(x, d=3):
    text = f"{x:.{d}f}"
    if float(text) == 0:
        text = f"{0:.{d}f}"
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def signed(x, d=3):
    text = f(x, d)
    return text if text.startswith("\\ensuremath{-}") or float(f"{x:.{d}f}") == 0 else "\\ensuremath{+}" + text


def ci(lo, hi, d=3):
    return f"[{signed(lo, d)}, {signed(hi, d)}]"


def sci(x, d=1):
    mant, exp = f"{x:.{d}e}".split("e")
    return f"\\ensuremath{{{mant}\\times 10^{{{int(exp)}}}}}"


def count(n):
    return f"{n:,}"


def pct(x, d=1):
    return f(100 * x, d)


def clock(utc):
    return datetime.fromisoformat(utc).strftime("%H:%M")


def clock_seconds(utc):
    return datetime.fromisoformat(utc).strftime("%H:%M:%S")


def mean(xs):
    return statistics.fmean(xs)


# ----------------------------------------------------------------------------------------------- design and setting
def design(n):
    conf = read(P2 / "protocol/confirmation.json")
    data = read(P2 / "data/confirmation.json")
    sched = data["schedules"]
    hist = read(AR / "posthoc-direction.json")["transition_counts"]
    n["NFamilies"] = count(len(conf["task_seeds"]))
    n["SeedFirst"] = str(conf["task_seeds"][0])
    n["SeedLast"] = str(conf["task_seeds"][-1])
    n["NInits"] = WORDS[len(conf["initialization_seeds"])]
    n["InitSeeds"] = " and ".join(map(str, conf["initialization_seeds"]))
    n["NModels"] = count(read(P2 / "runs/confirmation/manifest.json")["jobs"].__len__())
    n["NTrajectories"] = count(read(P2A / "confirmation/summary.json")["trajectories"])
    n["NArms"] = WORDS[len(conf["arms"])]
    n["Width"] = str(conf["width"])
    n["NWeights"] = str(16 * conf["width"] + conf["width"])
    n["NReadout"] = str(conf["width"])
    n["NActive"] = str(8 * conf["width"] + conf["width"])
    n["NSlotWeights"] = str(8 * conf["width"])
    n["BaseLR"] = str(conf["base_learning_rate"])
    n["UpdateLR"] = str(conf["update_learning_rate"])
    n["StepsPerEnv"] = str(conf["steps_per_environment"])
    n["FinalSteps"] = count(conf["steps_per_environment"] + conf["final_extra_steps"])
    total = conf["steps_per_environment"] * (len(sched["joint"]) - 1) + conf["steps_per_environment"] + conf["final_extra_steps"]
    n["BaseSteps"] = count(total)
    n["UpdateSteps"] = str(conf["update_steps"])
    n["NormStep"] = str(conf["normalized_step_size"])
    n["NormPath"] = f(conf["normalized_step_size"] * conf["update_steps"], 1)
    n["NCycles"] = WORDS[conf["cycles"]]
    n["NEnvVisits"] = str(len(sched["joint"]))
    n["NTransitions"] = str(len(sched["joint"]) - 1)
    n["NDistinctEnv"] = WORDS[len(set(sched["joint"]))]
    n["SuffixLength"] = WORDS[5]
    cycles, suffix = conf["cycles"], 5
    for history, key in (("independent", "IndCycle"), ("joint", "JointCycle")):
        order = sched[history]
        cycle = order[:8]
        assert order[:-suffix] == cycle * cycles and order[-suffix:] == sched["joint"][-suffix:]
        n[key] = ", ".join(cycle)
    n["SharedEnding"] = ", ".join(sched["joint"][-suffix:])
    n["IndFOnly"] = str(hist["independent"]["f_only"])
    n["IndGOnly"] = str(hist["independent"]["g_only"])
    n["IndBoth"] = str(hist["independent"]["both"])
    n["IndSingle"] = str(hist["independent"]["f_only"] + hist["independent"]["g_only"])
    times = {1: "once", 2: "twice"}
    n["JointFOnly"] = times.get(hist["joint"]["f_only"], f'{hist["joint"]["f_only"]} times')
    n["JointGOnly"] = times.get(hist["joint"]["g_only"], f'{hist["joint"]["g_only"]} times')
    n["JointBoth"] = str(hist["joint"]["both"])
    n["JointSingle"] = str(hist["joint"]["f_only"] + hist["joint"]["g_only"])
    checks = data["design_checks"]
    assert all(abs(c["transition_mse_min"] - 3.5) < 1e-9 and abs(c["transition_mse_max"] - 3.5) < 1e-9 for c in checks)
    n["TransitionSize"] = "3.5"
    n["TargetSquare"] = "1.75"
    n["NTrain"] = str(checks[0]["training_examples"])
    n["NWithheld"] = str(checks[0]["withheld_examples"])
    n["NObservedPairs"] = str(checks[0]["training_examples"] - 16)
    n["ThrPrimitive"] = f(conf["base_primitive_nrmse_max"], 2)
    n["ThrUnseen"] = f(conf["base_unseen_nrmse_max"], 2)
    n["MinQualified"] = WORDS[conf["minimum_qualified_development_pairs"]]
    n["Margin"] = f(conf["preservation_effect_margin"], 2)
    n["NoninfMargin"] = f(conf["adaptation_noninferiority_margin"], 2)
    n["TargetHalfWidth"] = f(conf["confirmation_target_half_width"], 3)
    n["MinFamilies"] = str(conf["confirmation_minimum_families"])
    n["MaxFamilies"] = str(conf["confirmation_maximum_families"])
    s = read(P2A / "confirmation/summary.json")
    n["PrimaryConfidence"] = f(100 * s["inference"]["primary_interval_confidence"], 2)
    n["DegreesFreedom"] = str(len(conf["task_seeds"]) - 1)
    n["NPrimaryIntervals"] = WORDS[s["inference"]["primary_interval_count"]]


# ----------------------------------------------------------------------------------------------- development
def development(n):
    dev = read(P2A / "development/summary.json")
    amd = read(P2A / "amendment/summary.json")
    n["NDevFamilies"] = WORDS[dev["families"]]
    n["NDevModels"] = str(dev["models"])
    n["NAmendModels"] = str(amd["models"])
    n["DevLinearPairs"] = f"{dev['base']['linear']['qualified_pairs']} of {dev['base']['linear']['total_pairs']}"
    n["DevReluPairs"] = f"{dev['base']['relu']['qualified_pairs']} of {dev['base']['relu']['total_pairs']}"
    n["DevLeakyPairs"] = f"{amd['base']['leaky']['qualified_pairs']} of {amd['base']['leaky']['total_pairs']}"
    n["DevReluUnseen"] = f(dev["base"]["relu"]["errors"]["combined_unseen"]["mean"], 3)
    n["DevReluUnseenMax"] = f(dev["base"]["relu"]["errors"]["combined_unseen"]["maximum"], 3)
    n["DevReluSeen"] = f(dev["base"]["relu"]["errors"]["combined_seen"]["mean"], 3)
    n["DevLinearEffect"] = signed(dev["primary"]["linear"]["effects"]["stable"]["mean"], 3)
    n["DevLeakyEffect"] = signed(amd["primary"]["leaky"]["effects"]["stable"]["mean"], 3)
    n["DevLinearSD"] = f(dev["primary"]["linear"]["effects"]["stable"]["sd"], 3)
    n["DevLeakySD"] = f(amd["primary"]["leaky"]["effects"]["stable"]["sd"], 3)
    n["DevLinearCI"] = ci(dev["primary"]["linear"]["effects"]["stable"]["low"], dev["primary"]["linear"]["effects"]["stable"]["high"], 2)
    n["DevLeakyCI"] = ci(amd["primary"]["leaky"]["effects"]["stable"]["low"], amd["primary"]["leaky"]["effects"]["stable"]["high"], 2)


# ----------------------------------------------------------------------------------------------- confirmation
def confirmation(n):
    s = read(P2A / "confirmation/summary.json")
    sens = read(P2A / "confirmation/descriptive-sensitivity.json")["results"]
    for arch in ARCHS:
        A = NAMES[arch]
        b = s["base"][arch]
        n[f"{A}QualifiedPairs"] = f"{b['qualified_pairs']} of {b['total_pairs']}"
        e = s["primary"][arch]["effects"]
        for metric, M in (("stable", "Stable"), ("changed", "Changed"), ("transfer", "Unseen")):
            x = e[metric]
            d = 5 if metric == "changed" else 3
            n[f"{A}{M}Ind"] = f(x["independent"]["mean"], d)
            n[f"{A}{M}Joint"] = f(x["joint"]["mean"], d)
            n[f"{A}{M}Diff"] = signed(x["mean"], d + (1 if metric == "changed" else 0))
            n[f"{A}{M}CI"] = ci(x["low"], x["high"], d + (1 if metric == "changed" else 0))
        n[f"{A}StableSD"] = f(e["stable"]["sd"], 3)
        n[f"{A}HalfWidth"] = f((e["stable"]["high"] - e["stable"]["low"]) / 2, 3)
        n[f"{A}FavorInd"] = str(sens[arch]["stable"]["families_favoring_independent"])
        n[f"{A}BootCI"] = ci(*sens[arch]["stable"]["percentile_interval_99_1667"], 3)
        n[f"{A}StableMin"] = signed(sens[arch]["stable"]["family_min"], 2)
        n[f"{A}StableMax"] = signed(sens[arch]["stable"]["family_max"], 2)
        for seed in ("2701", "2702"):
            v = sens[arch][f"initialization_{seed}"]
            n[f"{A}Init{'A' if seed == '2701' else 'B'}"] = signed(v["mean"], 3)
            n[f"{A}Init{'A' if seed == '2701' else 'B'}CI"] = ci(*v["descriptive_95_interval"], 3)
        n[f"{A}Decision"] = {"inconclusive": "inconclusive",
                             "practical_independent_history_benefit_ruled_out": "benefit excluded"}[s["primary"][arch]["decision"]]
    stable = [s["primary"][a]["effects"]["stable"][h]["mean"] for a in ARCHS for h in ("independent", "joint")]
    n["EffectLowest"] = signed(min(s["primary"][a]["effects"]["stable"]["low"] for a in ARCHS), 3)
    n["EffectHighest"] = signed(max(s["primary"][a]["effects"]["stable"]["high"] for a in ARCHS), 3)
    n["StableAllLow"] = f(min(stable), 2)
    n["StableAllHigh"] = f(max(stable), 2)
    n["StableAll"] = f(mean(stable), 2)
    changed = [s["primary"][a]["effects"]["changed"][h]["mean"] for a in ARCHS for h in ("independent", "joint")]
    n["ChangedAll"] = f(mean(changed), 4)
    n["ChangedDiffBound"] = f(max(max(abs(s["primary"][a]["effects"]["changed"]["low"]),
                                      abs(s["primary"][a]["effects"]["changed"]["high"])) for a in ARCHS), 5)
    n["ChangedAllMax"] = f(max(changed), 4)
    n["LeakyStableLow"] = signed(s["primary"]["leaky"]["effects"]["stable"]["low"], 3)
    leaky_margin = -0.05 - s["primary"]["leaky"]["effects"]["stable"]["low"]
    n["LeakyExclusionGap"] = f(-leaky_margin, 4)
    n["IdentityMaxErr"] = sci(s["identity_max_error"], 1)
    hw = [(s["primary"][a]["effects"]["stable"]["high"] - s["primary"][a]["effects"]["stable"]["low"]) / 2 for a in ARCHS]
    n["HalfWidthLow"] = f(min(hw), 3)
    n["HalfWidthHigh"] = f(max(hw), 3)
    n["NBootstrap"] = count(read(P2A / "confirmation/descriptive-sensitivity.json")["resamples"])
    for arch in ARCHS:
        b = s["base"][arch]["errors"]
        A = NAMES[arch]
        for k, K in (("f", "F"), ("g", "G"), ("combined_seen", "Seen"), ("combined_unseen", "Unseen")):
            n[f"{A}Base{K}"] = f(b[k]["mean"], 4)
            n[f"{A}Base{K}Max"] = f(b[k]["maximum"], 4)
    lin_max = max(s["base"]["linear"]["errors"][k]["maximum"] for k in ("f", "g", "combined_seen", "combined_unseen"))
    n["LinearBaseMaxAll"] = sci(lin_max, 0)


def arms(n):
    """Update conditions at step 200 (histories pooled, family means) and progress-matched results."""
    s = read(P2A / "confirmation/summary.json")
    d = s["descriptive"]
    pooled = lambda a, arm, m: mean([d[a][h][arm][m]["mean"] for h in ("independent", "joint")])
    for arch in ARCHS:
        A = NAMES[arch]
        n[f"{A}FreezeChanged"] = f(pooled(arch, "freeze_readout", "changed"), 3)
        n[f"{A}FreezeStable"] = f(pooled(arch, "freeze_readout", "stable"), 4)
        assert all(d[arch][h]["freeze_readout"]["stable_drift"]["mean"] == 0 for h in ("independent", "joint"))
        rnd = [pooled(arch, f"random_freeze_{i}", "stable") for i in range(3)]
        n[f"{A}RandomStable"] = f(mean(rnd), 2)
        n[f"{A}ReadoutOnlyStable"] = f(pooled(arch, "readout_only", "stable"), 2)
        n[f"{A}ReadoutOnlyChanged"] = f(pooled(arch, "readout_only", "changed"), 2)
        n[f"{A}UnrestrictedPath"] = f(pooled(arch, "unrestricted", "path"), 2)
        n[f"{A}FreezePath"] = f(pooled(arch, "freeze_readout", "path"), 2)
        prog = s["matched_progress"][arch]
        n[f"{A}ProgressSteps"] = f(mean([prog[h]["unrestricted"]["step_mean"] for h in ("independent", "joint")]), 0)
        n[f"{A}ProgressStable"] = f(mean([prog[h]["unrestricted"]["stable_mean"] for h in ("independent", "joint")]), 2)
        n[f"{A}NormReached"] = str(sum(prog[h]["normalized"]["reached"] for h in ("independent", "joint")))
        ctl = s["readout_controls"][arch]
        n[f"{A}FreezeMinusRandom"] = signed(mean([ctl[h]["freeze_minus_mean_random"]["mean"] for h in ("independent", "joint")]), 2)
    allrandom = [pooled(a, f"random_freeze_{i}", "stable") for a in ARCHS for i in range(3)]
    n["RandomStableAll"] = f(mean(allrandom), 2)
    n["ReadoutOnlyStableAll"] = f(mean([pooled(a, "readout_only", "stable") for a in ARCHS]), 2)
    fr = [prog["freeze_readout"]["reached"] for a in ARCHS for prog in s["matched_progress"][a].values()]
    tot = [prog["freeze_readout"]["total"] for a in ARCHS for prog in s["matched_progress"][a].values()]
    n["FreezeReached"] = str(sum(fr))
    n["FreezeTotal"] = count(sum(tot))
    steps = [s["matched_progress"][a][h]["unrestricted"]["step_mean"] for a in ARCHS for h in ("independent", "joint")]
    n["ProgressStepsLow"] = f(min(steps), 0)
    n["ProgressStepsHigh"] = f(max(steps), 0)
    st = [s["matched_progress"][a][h]["unrestricted"]["stable_mean"] for a in ARCHS for h in ("independent", "joint")]
    n["ProgressStableLow"] = f(min(st), 2)
    n["ProgressStableHigh"] = f(max(st), 2)
    n["OldLabelChanged"] = f(math.sqrt(2), 2)
    # Fixed-length steps (normalized-gradient arm): descriptive 95% intervals.
    for arch in ARCHS:
        A = NAMES[arch]
        ns = s["normalized_sensitivity"][arch]
        n[f"{A}NormChangedDiff"] = signed(ns["changed"]["mean"], 3)
        n[f"{A}NormChangedCI"] = ci(ns["changed"]["low"], ns["changed"]["high"], 3)
        n[f"{A}NormStableDiff"] = signed(ns["stable"]["mean"], 3)
        n[f"{A}NormStableCI"] = ci(ns["stable"]["low"], ns["stable"]["high"], 3)
        assert ns["path"]["mean"] == 0 and ns["path"]["sd"] == 0
    norm_changed = [-s["normalized_sensitivity"][a]["changed"]["mean"] for a in ARCHS]
    n["NormChangedLow"] = f(min(norm_changed), 3)
    n["NormChangedHigh"] = f(max(norm_changed), 3)
    n["NormZeroSteps"] = WORDS[s["normalized_zero_gradient_steps"]] if s["normalized_zero_gradient_steps"] < 10 else str(s["normalized_zero_gradient_steps"])


def duration(n):
    d = read(P2 / "runs/duration-control/summary.json")
    c = read(AR / "duration-curves.json")
    n["DurSteps"] = count(d["steps"])
    n["DurTrajectories"] = count(d["trajectories"])
    fp = c["first_passage"]
    n["DurMaxTen"] = str(max(v["max_steps_0_10"] for v in fp.values()))
    n["DurMaxOne"] = str(max(v["max_steps_0_01"] for v in fp.values()))
    n["DurReached"] = count(sum(v["reached_0_01"] for v in fp.values()))
    for arch in ARCHS:
        A = NAMES[arch]
        n[f"{A}DurTen"] = f(fp[arch]["mean_steps_0_10"], 0)
        n[f"{A}DurOne"] = f(fp[arch]["mean_steps_0_01"], 0)
        n[f"{A}Slowdown"] = f(fp[arch]["slowdown_0_10"], 1)
        n[f"{A}DurChanged"] = sci(fp[arch]["endpoint_changed_mean"], 1)
    n["DurChangedWorst"] = f(max(v["endpoint_changed_max"] for v in fp.values()), 4)
    n["DurUnseenWorst"] = f(max(v["endpoint_transfer_max"] for v in fp.values()), 3)
    assert max(v["endpoint_stable_drift_max"] for v in fp.values()) == 0
    n["DurReplays"] = str(d["independent_torch_replayed_trajectories"])
    n["DurTorchErr"] = sci(d["torch_vs_numpy_1000_max_error"], 1)
    n["DurSeconds"] = f(d["seconds"], 1)
    sl = [fp[a]["slowdown_0_10"] for a in ARCHS]
    n["SlowdownLow"] = f(min(sl), 1)
    n["SlowdownHigh"] = f(max(sl), 1)


# ----------------------------------------------------------------------------------------------- post hoc
def posthoc(n):
    e = read(AR / "posthoc-endpoints.json")["architectures"]
    for arch in ARCHS:
        A = NAMES[arch]
        n[f"{A}TwoWayCI"] = ci(e[arch]["two_way"]["low"], e[arch]["two_way"]["high"], 3)
        u = e[arch]["unseen_vs_unchanged"]
        n[f"{A}UnseenCorr"] = f(u["correlation"], 7 if arch == "linear" else 4)
        n[f"{A}UnseenDev"] = f(u["max_abs_deviation_from_unchanged_over_sqrt2"], 3)
        n[f"{A}UnseenFamilyCorr"] = f(u["family_difference_correlation"], 4)
        n[f"{A}UnseenSlope"] = f(u["family_difference_slope"], 3)
    n["TwoWayConfidence"] = f(100 * e["linear"]["two_way"]["confidence"], 1)
    assert read(AR / "posthoc-endpoints.json")["conclusions_unchanged"]
    dirn = read(AR / "posthoc-direction.json")["effects"]
    for arch in ARCHS:
        A = NAMES[arch]
        for b, B in (("f", "F"), ("g", "G"), ("f_minus_g", "Inter")):
            x = dirn[arch]["unrestricted"][b]
            n[f"{A}Dir{B}"] = signed(x["mean"], 3)
            n[f"{A}Dir{B}CI"] = ci(x["low"], x["high"], 3)
    ss = read(AR / "posthoc-sample-size.json")
    n["RuleFamilies"] = count(ss["rule_families_uncapped"])
    n["RuleMultiplier"] = f(ss["rule_multiplier"], 1)
    n["RuleSD"] = f(ss["rule_sd"], 3)
    n["TQuantile"] = f(ss["t_quantile_six_way"], 2)
    n["TQuantileTwo"] = f(ss["t_quantile_two_way"], 2)
    n["RuleFamiliesT"] = count(ss["rule_families_with_t_quantile"])
    n["ExpectedHalfWidthDev"] = f(ss["expected_half_width_at_cap_development_sd"], 3)
    n["ExpectedHalfWidthConf"] = f(ss["expected_half_width_at_cap_confirmation_sd"], 3)
    n["ProbInsideSix"] = pct(ss["probability_inside_0_05_at_64_six_way"], 1)
    n["ProbInsideTwo"] = pct(ss["probability_inside_0_05_at_64_two_way"], 0)
    n["SigmaConf"] = f(ss["sigma_used"], 3)
    need = ss["families_needed"]
    n["NeedFiveTwo"] = str(need["0.05"]["2_way"]["power_0_8"])
    n["NeedQuarterTwo"] = str(need["0.025"]["2_way"]["power_0_8"])
    n["NeedFiveSix"] = str(need["0.05"]["6_way"]["power_0_8"])
    n["NeedQuarterSix"] = str(need["0.025"]["6_way"]["power_0_8"])
    n["HalfFiveTwo"] = str(need["0.05"]["2_way"]["expected_half_width_equals_margin"])
    n["HalfQuarterTwo"] = str(need["0.025"]["2_way"]["expected_half_width_equals_margin"])
    n["HalfFiveSix"] = str(need["0.05"]["6_way"]["expected_half_width_equals_margin"])
    n["HalfQuarterSix"] = str(need["0.025"]["6_way"]["expected_half_width_equals_margin"])
    p = ss["probability_inside_two_way"]
    n["ProbAtOneTen"] = pct(p["0.05@110"], 0)
    n["ProbAtThreeFifty"] = pct(p["0.025@350"], 0)
    n["NeedFiveRound"] = str(int(round(need["0.05"]["2_way"]["power_0_8"], -1)))
    n["NeedQuarterRound"] = str(int(round(need["0.025"]["2_way"]["power_0_8"], -2)))
    g = read(AR / "posthoc-geometry.json")["architectures"]
    for arch in ARCHS:
        A = NAMES[arch]
        n[f"{A}PairDistance"] = f(g[arch]["distance_over_norm"]["median"], 2)
        n[f"{A}PairDistanceRange"] = f"{f(g[arch]['distance_over_norm']['min'], 2)}--{f(g[arch]['distance_over_norm']['max'], 2)}"
        n[f"{A}OverlapCorr"] = signed(g[arch]["correlation_with_damage"]["code_overlap"], 2)
        n[f"{A}OverlapPaired"] = signed(g[arch]["paired_difference_correlation"]["code_overlap"], 2)
    n["GeometryMaxNetworkCorr"] = f(max(abs(v) for a in ARCHS for v in g[a]["correlation_with_damage"].values()), 2)
    n["GeometryMaxVariance"] = f(100 * max(v * v for a in ARCHS for v in g[a]["correlation_with_damage"].values()), 0)
    n["GeometryFamilies"] = str(g["linear"]["families"])
    for arch in ARCHS:
        n[f"{NAMES[arch]}OverlapPairedP"] = f(g[arch]["paired_difference_p"]["code_overlap"], 3)
        n[f"{NAMES[arch]}OverlapPairedVar"] = f(100 * g[arch]["paired_difference_correlation"]["code_overlap"] ** 2, 0)
    top = max(((a, k) for a in ARCHS for k in g[a]["paired_difference_correlation"]),
              key=lambda ak: abs(g[ak[0]]["paired_difference_correlation"][ak[1]]))
    assert top == ("linear", "code_overlap")
    paired = sum(len(g[a]["paired_difference_correlation"]) for a in ARCHS)      # 3 measures x 2 architectures
    n["GeometryNCorrelations"] = WORDS.get(paired, str(paired))
    n["GeometryMaxPairedCorr"] = f(max(abs(v) for a in ARCHS for v in g[a]["paired_difference_correlation"].values()), 2)
    fr = read(AR / "posthoc-frozen-rate.json")["architectures"]
    n["FrozenExactLinear"] = f"{fr['linear']['exact_0_10']} of {fr['linear']['trajectories']}"
    n["FrozenLinearCount"] = str(fr["linear"]["trajectories"])
    assert fr["linear"]["exact_0_01"] == fr["linear"]["trajectories"] == fr["linear"]["exact_0_10"]
    n["LeakyFrozenMaxErr"] = str(fr["leaky"]["max_abs_error_0_10"])
    n["LeakyFrozenMaxErrOne"] = str(fr["leaky"]["max_abs_error_0_01"])
    n["LeakyFrozenMeanErr"] = f(fr["leaky"]["mean_abs_error_0_10"], 1)
    n["ReadoutGain"] = f(fr["linear"]["mean_gain"], 2)
    n["CodeNormSq"] = f(fr["linear"]["mean_code_norm_squared"], 1)
    n["CodeRatio"] = f(fr["linear"]["code_to_readout_ratio"], 1)
    n["PairDistanceLow"] = f(min(g[a]["distance_over_norm"]["median"] for a in ARCHS), 1)
    n["PairDistanceHigh"] = f(max(g[a]["distance_over_norm"]["median"] for a in ARCHS), 1)


# ----------------------------------------------------------------------------------------------- phase 1
def phase1(n):
    cfg = read(P1 / "03-Data/config/execution-v1.json")
    lng = read(P1 / "03-Data/config/long-final-v1.json")
    pilot = read(P1R / "pilot-analysis-v1/summary.json")
    longf = read(P1R / "long-final-analysis-v1/summary.json")
    ref = read(P1R / "modular-reference-v1/summary.json")
    h1, h2 = cfg["hidden_sizes"]
    import re
    inputs = int(re.search(r"torch\.nn\.Linear\((\d+), sizes\[0\]\)", (P1 / "02-Code/experiment.py").read_text()).group(1))
    n["POneInputs"] = str(inputs)
    n["POneHidden"] = str(h1)
    n["POneParams"] = count(inputs * h1 + h1 + h1 * h2 + h2 + h2 * 8 + 8)
    n["POneLR"] = str(cfg["learning_rate"])
    n["POneFamilies"] = WORDS[len(cfg["task_seeds"])]
    n["POneModels"] = str(pilot["completed_jobs"])
    orders = {len(s["order"]) for s in read(P1 / "03-Data/prepared/training-schedules.json")}
    assert len(orders) == 1
    n["POneEnvironments"] = str(orders.pop())
    n["POneSteps"] = count(cfg["steps_per_environment"] * int(n["POneEnvironments"]))
    n["POneFinalLong"] = count(lng["steps_per_environment"] + lng["final_extra_steps"])
    factor = (lng["steps_per_environment"] + lng["final_extra_steps"]) // cfg["steps_per_environment"]
    n["POneLongFactor"] = {10: "ten"}.get(factor, str(factor))
    names = {"factorwise": "Factorwise", "mixed_change": "Mixed", "shuffled": "Shuffled"}
    for h, H in names.items():
        n[f"POne{H}Unseen"] = pct(pilot["base_history_means"][h]["combined_heldout"], 1)
        n[f"POne{H}UnseenLong"] = pct(longf["base_history_means"][h]["combined_heldout"], 1)
    u = [pilot["base_history_means"][h]["combined_heldout"] for h in names]
    ul = [longf["base_history_means"][h]["combined_heldout"] for h in names]
    n["POneUnseenLow"] = pct(min(u), 1)
    n["POneUnseenHigh"] = pct(max(u), 1)
    n["POneLongLow"] = pct(min(ul), 1)
    n["POneLongHigh"] = pct(max(ul), 1)
    n["POneChance"] = pct(1 / 8, 1)
    n["POneThreshold"] = pct(cfg["gate"]["base_heldout_combined_accuracy"], 0)
    n["POneSeenThreshold"] = pct(cfg["gate"]["base_seen_combined_accuracy"], 0)
    n["POneStepsPerEnv"] = str(cfg["steps_per_environment"])
    n["POneUpdateSteps"] = str(max(cfg["evaluation_steps"]))
    import csv
    rows = list(csv.DictReader(open(P1R / "pilot-analysis-v1/base.csv")))
    n["POneBest"] = pct(max(float(r["combined_heldout"]) for r in rows), 0)
    fe = [r for r in pilot["family_endpoints"] if r["optimizer_mode"] == "reset" and r["condition"] == "changed_primitive"]
    for key, K in (("unchanged_primitive", "Unchanged"), ("combined_heldout", "UpdUnseen")):
        vals = [mean([r[key] for r in fe if r["history"] == h]) for h in names]
        n[f"POne{K}Low"] = pct(min(vals), 1)
        n[f"POne{K}High"] = pct(max(vals), 1)
    fr = [r for r in pilot["family_endpoints"] if r["optimizer_mode"] == "retained" and r["condition"] == "changed_primitive"]
    vals = [mean([r["unchanged_primitive"] for r in fr if r["history"] == h]) for h in names]
    n["POneRetainedLow"] = pct(min(vals), 1)
    n["POneRetainedHigh"] = pct(max(vals), 1)
    n["POneRefModels"] = WORDS[len(ref["models"])]
    n["POneEffect"] = f(100 * pilot["mean_primary_history_effect"], 2)
    n["POneEffectSD"] = f(100 * pilot["within_family_initialization_sd"], 2)
    n["POneEffectThreshold"] = f(100 * cfg["gate"]["minimum_factorwise_advantage"], 0)
    n["POneAuditRecords"] = count(read(P2 / "RELEASE-VERIFICATION.json")["original_study_verification"]["audited_raw_records"])
    n["POneFiles"] = count(read(P2 / "RELEASE-VERIFICATION.json")["original_study_verification"]["files_checked"])
    n["POneTrajectories"] = str(read(P1 / "06-Study-History/STATUS.json")["pilot_update_and_control_trajectories"])


# ----------------------------------------------------------------------------------------------- verification
def verification(n):
    rv = read(P2 / "RELEASE-VERIFICATION.json")
    n["AuditRecords"] = count(rv["audited_per_example_records"])
    n["AuditCheckpoints"] = count(rv["audited_checkpoints"])
    n["NumpyReplays"] = str(rv["independent_numpy_trajectories_replayed"])
    n["PTwoFiles"] = count(rv["files_checked"])
    audits = [read(P2A / f"{s}-audit.json") for s in ("development", "amendment", "confirmation")]
    n["NumpyReplayErr"] = sci(max(a["numpy_replay_max_error"] for a in audits), 1)
    rep = read(P2 / "runs/selected-replay/verification.json")
    n["ReplayModels"] = WORDS[rep["models"]]
    n["ReplayArrays"] = count(rep["array_comparisons"])
    n["ConfMinutes"] = f(read(P2 / "runs/confirmation/status.json")["seconds"] / 60, 0)
    n["ConfMemory"] = str(round(read(P2 / "runs/confirmation/status.json")["maximum_rss_kib"] / 1024))
    inventory = (PROJECT / "provenance/relocation-inventory-2026-10-09.tsv").read_text().splitlines()
    rows = [line.split("\t") for line in inventory if line and not line.startswith("#")][1:]
    assert all(r[2] == r[3] for r in rows)
    n["RelocatedFiles"] = count(len(rows))
    replays = sorted((PROJECT / "analysis/results").glob("replay-*.json"))
    if replays:
        r = read(replays[-1])
        assert r["status"] == "passed"
        n["NewReplayModels"] = WORDS[r["phase2_models"]["models"]]
        n["NewReplayArrays"] = count(r["phase2_models"]["array_comparisons"])
        n["NewReplayTorch"] = r["environment"]["torch"].replace("+", "{+}")
        n["NewReplayDate"] = datetime.fromisoformat(r["created_utc"]).strftime("%-d %B %Y")
        assert all(v["status"] == "passed" and not v["differing_fields"] for v in r["phase2_audits"].values())


def timeline(n):
    p1 = P1 / "06-Study-History"
    n["TimeOneExecution"] = clock(read(p1 / "EXECUTION-FREEZE-v1.json")["created_utc"])
    n["TimeOneDiagnostic"] = clock(read(p1 / "DIAGNOSTIC-FREEZE-v1.json")["created_utc"])
    n["TimeOneAnalysis"] = clock(read(p1 / "ANALYSIS-FREEZE-v1.json")["created_utc"])
    n["TimeOnePilot"] = clock(read(P1R / "pilot-v1/manifest.json")["created_utc"])
    n["TimeOneLong"] = clock(read(P1R / "long-final-v1/manifest.json")["created_utc"])
    n["TimeOneReference"] = clock(read(P1R / "modular-reference-v1/manifest.json")["created_utc"])
    n["TimeOneManifest"] = clock(read(p1 / "FINAL-EVIDENCE-MANIFEST.json")["created_utc"])
    n["TimeDevStart"] = clock(read(P2 / "runs/development/manifest.json")["created_utc"])
    n["TimeAmendStart"] = clock(read(P2 / "runs/amendment/manifest.json")["created_utc"])
    n["TimeReplay"] = clock(read(P2 / "runs/selected-replay/verification.json")["completed_utc"])
    n["TimeDurationRun"] = clock(read(P2 / "runs/duration-control/summary.json")["created_utc"])
    for name, key in (("DEVELOPMENT", "Development"), ("AMENDMENT", "Amendment"), ("CONFIRMATION", "Confirmation"),
                      ("VERIFICATION", "Verification"), ("DISPLAY", "Display"), ("DURATION", "Duration")):
        n[f"Time{key}"] = clock(read(P2 / f"protocol/{name}-FREEZE.json")["created_utc"])
    n["TimeConfTasks"] = clock(read(P2 / "data/confirmation.json")["created_utc"])
    n["TimeConfStart"] = clock(read(P2 / "runs/confirmation/manifest.json")["created_utc"])
    n["TimeConfirmationSec"] = clock_seconds(read(P2 / "protocol/CONFIRMATION-FREEZE.json")["created_utc"])
    n["TimeConfTasksSec"] = clock_seconds(read(P2 / "data/confirmation.json")["created_utc"])
    n["TimeConfStartSec"] = clock_seconds(read(P2 / "runs/confirmation/manifest.json")["created_utc"])
    span = (datetime.fromisoformat(read(P2 / "FINAL-EVIDENCE-MANIFEST.json")["created_utc"])
            - datetime.fromisoformat(read(P2 / "protocol/DEVELOPMENT-FREEZE.json")["created_utc"])).total_seconds() / 60
    n["PhaseTwoMinutes"] = str(round(span))
    n["TimeConfEnd"] = clock(read(P2 / "runs/confirmation/status.json")["finished_utc"])
    n["TimeConfAnalysis"] = clock(read(P2A / "confirmation/provenance.json")["created_utc"])
    n["TimeDevAnalysis"] = clock(read(P2A / "development/provenance.json")["created_utc"])
    n["TimeAmendAnalysis"] = clock(read(P2A / "amendment/provenance.json")["created_utc"])
    n["TimeManifest"] = clock(read(P2 / "FINAL-EVIDENCE-MANIFEST.json")["created_utc"])


# ----------------------------------------------------------------------------------------------- study 3
P3 = PROJECT / "studies/3-larger-confirmation"


def study3(n):
    r = read(P3 / "results/summary.json")
    v = read(P3 / "results/verification.json")
    fr = read(P3 / "FREEZE.json")
    st = read(P3 / "runs/confirmation/status.json")
    man = read(P3 / "runs/confirmation/manifest.json")
    ss = read(P3 / "protocol/sample-size.json")
    sc = read(P3 / "protocol/seed-check.json")
    cfg = read(P3 / "protocol/config.json")
    assert v["status"] == "passed" and st["status"] == "complete" and not st["failures"]
    assert fr["created_utc"] < man["created_utc"]
    n["ThreeFamilies"] = str(r["families"])
    n["ThreeNetworks"] = count(r["networks"])
    n["ThreeTrajectories"] = count(r["trajectories"])
    n["ThreeSeedFirst"], n["ThreeSeedLast"] = map(str, r["task_seeds"])
    n["ThreeInitSeeds"] = " and ".join(map(str, r["initialization_seeds"]))
    n["ThreeEarlierSeeds"] = str(sc["earlier_seed_count"])
    n["ThreeFreezeTime"] = clock(fr["created_utc"])
    n["ThreeRunStart"] = clock(man["created_utc"])
    n["ThreeRunEnd"] = clock(st["finished_utc"])
    n["ThreeRunMinutes"] = f(st["seconds"] / 60, 0)
    n["ThreeCPUMinutes"] = f(st["job_seconds"] / 60, 0)
    n["ThreeWorkers"] = WORDS.get(st["workers"], str(st["workers"]))
    n["ThreeConfidence"] = f(100 * r["primary_confidence"], 1)
    n["ThreePlanSD"] = f(ss["sigma_used"], 3)
    n["ThreePlanHalfWidth"] = f(ss["expected_half_width_at_n"], 3)
    n["ThreePlanProbLinear"] = f(ss["probability_inside_at_n"]["linear"], 2)
    n["ThreePlanProbLeaky"] = f(ss["probability_inside_at_n"]["leaky"], 2)
    n["ThreePlanProbBoth"] = f(ss["probability_both_inside_if_independent"], 2)
    assert ss["families"] == r["families"] == len(cfg["task_seeds"])
    comp = {a: r["competence"][a] for a in ARCHS}
    assert comp["linear"]["competent_pairs"] == comp["linear"]["pairs"] and comp["leaky"]["competent_pairs"] == comp["leaky"]["pairs"]
    n["ThreePairs"] = str(comp["linear"]["pairs"])
    words = {"inconclusive": "inconclusive", "equivalence_established": "equivalence established"}
    for arch in ARCHS:
        A = NAMES[arch]
        p, sec = r["primary"][arch], r["secondary"][arch]
        n[f"{A}ThreeDiff"] = signed(p["mean"], 3)
        n[f"{A}ThreeCI"] = ci(p["low"], p["high"], 3)
        n[f"{A}ThreeLow"] = signed(p["low"], 3)
        n[f"{A}ThreeHigh"] = signed(p["high"], 3)
        n[f"{A}ThreeExcluded"] = f(-p["low"], 3)
        n[f"{A}ThreeOver"] = f(p["high"] - MARGIN_P, 3)
        n[f"{A}ThreeSD"] = f(p["sd"], 3)
        n[f"{A}ThreeHalfWidth"] = f(p["half_width"], 3)
        n[f"{A}ThreeDecision"] = words.get(p["decision"], p["decision"].replace("_", " "))
        n[f"{A}ThreeFavorInd"] = str(p["families_favouring_independent"])
        n[f"{A}ThreeInd"] = f(p["independent_mean"], 3)
        n[f"{A}ThreeJoint"] = f(p["joint_mean"], 3)
        for key, K in (("changed", "Changed"), ("transfer", "Unseen")):
            d = 6 if key == "changed" else 3
            n[f"{A}Three{K}Diff"] = signed(sec[key]["mean"], d)
            n[f"{A}Three{K}CI"] = ci(sec[key]["low"], sec[key]["high"], d)
        for key, K in (("f", "DirF"), ("g", "DirG"), ("f_minus_g", "DirInter")):
            n[f"{A}Three{K}"] = signed(sec["direction"][key]["mean"], 3)
            n[f"{A}Three{K}CI"] = ci(sec["direction"][key]["low"], sec["direction"][key]["high"], 3)
        for key, K in (("changed", "NormChanged"), ("stable", "NormStable")):
            n[f"{A}Three{K}"] = signed(sec["fixed_length"][key]["mean"], 3)
            n[f"{A}Three{K}CI"] = ci(sec["fixed_length"][key]["low"], sec["fixed_length"][key]["high"], 3)
        n[f"{A}ThreePooled"] = signed(sec["pooled_with_phase2"]["mean"], 3)
        n[f"{A}ThreePooledCI"] = ci(sec["pooled_with_phase2"]["low"], sec["pooled_with_phase2"]["high"], 3)
        n[f"{A}ThreeProgress"] = f(sec["progress_unrestricted"]["mean_step"], 0)
        n[f"{A}ThreeRandom"] = f(mean([sec["conditions"][f"random_freeze_{i}"]["stable"] for i in range(3)]), 2)
        n[f"{A}ThreeReadoutOnly"] = f(sec["conditions"]["readout_only"]["stable"], 2)
        assert sec["conditions"]["freeze_readout"]["stable_drift_max"] == 0
        assert sec["conditions"]["freeze_readout"]["reached_0_10"] == 0
        assert decide_pooled(sec["pooled_with_phase2"]) == "inside"
    n["ThreePooledFamilies"] = str(r["secondary"]["linear"]["pooled_with_phase2"]["n"])
    n["ThreeIdentityErr"] = sci(r["identity_max_error"], 1)
    n["ThreeChangedAll"] = f(mean([r["secondary"][a]["conditions"]["unrestricted"]["changed"] for a in ARCHS]), 4)
    dmg = [r["primary"][a][k] for a in ARCHS for k in ("independent_mean", "joint_mean")]
    n["ThreeStableLow"], n["ThreeStableHigh"] = f(min(dmg), 2), f(max(dmg), 2)
    n["ThreePredHeld"] = str(r["predictions_held"])
    n["ThreePredTotal"] = str(r["predictions_total"])
    n["ThreeVerifyComparisons"] = str(v["comparisons"])
    n["ThreeManifestFiles"] = count(len(read(P3 / "EVIDENCE-MANIFEST.json")["files"]))
    n["ThreeVerifyMaxErr"] = sci(max(v["maxima"].values()), 1)
    assert all(r["phase2_rule"]["decisions"][a] == "practical_independent_history_benefit_ruled_out" for a in ARCHS)


MARGIN_P = 0.05


def decide_pooled(iv):
    return "inside" if -MARGIN_P < iv["low"] and iv["high"] < MARGIN_P else "outside"


def numbers():
    n = {}
    for fn in (design, development, confirmation, arms, duration, posthoc, phase1, verification, timeline, study3):
        fn(n)
    return n


# ----------------------------------------------------------------------------------------------- tables
def tables():
    out = {}
    data = read(P2 / "data/confirmation.json")
    conf = read(P2 / "protocol/confirmation.json")
    hist = read(AR / "posthoc-direction.json")["transition_counts"]
    sched = data["schedules"]
    steps = conf["steps_per_environment"] * len(sched["joint"]) + conf["final_extra_steps"]
    same = lambda v: f"{v} & {v}"
    rows = [
        ("Environments visited (eight distinct, same visit counts)", same(len(sched["joint"]))),
        ("First environment; last five environments", "\\multicolumn{2}{c}{identical}"),
        ("Base training steps (Adam)", same(count(steps))),
        ("Mean squared target, every environment", same("1.75")),
        ("Mean squared target change, every transition", same("3.5")),
        ("Initial weights", "\\multicolumn{2}{c}{shared within each pair}"),
        ("Transitions changing $f$ only", f"{hist['independent']['f_only']} & {hist['joint']['f_only']}"),
        ("Transitions changing $g$ only", f"{hist['independent']['g_only']} & {hist['joint']['g_only']}"),
        ("Transitions changing both", f"{hist['independent']['both']} & {hist['joint']['both']}"),
    ]
    out["table-matching.tex"] = "\n".join(f"{a} & {b} \\\\" + (" \\midrule" if i == 5 else "") for i, (a, b) in enumerate(rows)) + "\n"

    s = read(P2A / "confirmation/summary.json")
    dev = read(P2A / "development/summary.json")
    amd = read(P2A / "amendment/summary.json")
    lines = []
    for label, block, arch in (("Linear", s, "linear"), ("Leaky ReLU", s, "leaky")):
        b = block["base"][arch]
        e = b["errors"]
        lines.append(f"{label} & {b['qualified_pairs']}/{b['total_pairs']} & " + " & ".join(
            f(e[k]["mean"], 4) for k in ("f", "g", "combined_seen", "combined_unseen")) + f" & {f(e['combined_unseen']['maximum'], 4)} \\\\")
    out["table-competence.tex"] = "\n".join(lines) + "\n"

    lines = []
    for label, block, arch, n_ in (("Linear", dev, "linear", "development"), ("ReLU", dev, "relu", "development"),
                                    ("Leaky ReLU", amd, "leaky", "amendment")):
        b = block["base"][arch]
        e = b["errors"]
        eff = block["primary"][arch]["effects"]["stable"]
        lines.append(f"{label} & {n_} & {b['qualified_pairs']}/{b['total_pairs']} & " + " & ".join(
            f(e[k]["mean"], 3) for k in ("f", "g", "combined_seen", "combined_unseen")) +
            f" & {signed(eff['mean'], 3)} & {f(eff['sd'], 3)} \\\\")
    out["table-development.tex"] = "\n".join(lines) + "\n"

    posth = read(AR / "posthoc-endpoints.json")["architectures"]
    lines = []
    for arch, label in (("linear", "Linear"), ("leaky", "Leaky ReLU")):
        e = s["primary"][arch]["effects"]
        for metric, mlabel, d in (("stable", "Unchanged skill", 3), ("changed", "Changed skill", 5), ("transfer", "Withheld pairs", 3)):
            x = e[metric]
            two = ci(posth[arch]["two_way"]["low"], posth[arch]["two_way"]["high"], 3) if metric == "stable" else "--"
            dd = d + (1 if metric == "changed" else 0)
            lab = f"\\multirow{{3}}{{*}}{{{label}}}" if metric == "stable" else ""
            lines.append(f"{lab} & {mlabel} & {f(x['independent']['mean'], d)} & {f(x['joint']['mean'], d)} & "
                         f"{signed(x['mean'], dd)} & {ci(x['low'], x['high'], dd)} & {two} \\\\")
        if arch == "linear":
            lines.append("\\midrule")
    out["table-primary.tex"] = "\n".join(lines) + "\n"

    # All update conditions at step 200 (family means; both histories shown).
    order = [("unrestricted", "Unrestricted"), ("freeze_readout", "Readout frozen"), ("random_freeze_0", "Random mask 1"),
             ("random_freeze_1", "Random mask 2"), ("random_freeze_2", "Random mask 3"), ("readout_only", "Readout only"),
             ("normalized", "Fixed-length steps"), ("no_update", "No update"), ("old_primitive", "Old values")]
    lines = []
    for arch, label in (("linear", "Linear"), ("leaky", "Leaky ReLU")):
        d = s["descriptive"][arch]
        prog = s["matched_progress"][arch]
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
        for arm, alab in order:
            cells = []
            for m in ("changed", "stable", "transfer"):
                cells.append(f"{f(d['independent'][arm][m]['mean'], 3)} / {f(d['joint'][arm][m]['mean'], 3)}")
            reached = prog["independent"][arm]["reached"] + prog["joint"][arm]["reached"]
            total = prog["independent"][arm]["total"] + prog["joint"][arm]["total"]
            path = mean([d[h][arm]["path"]["mean"] for h in ("independent", "joint")])
            lines.append(f"\\quad {alab} & " + " & ".join(cells) + f" & {reached}/{total} & {f(path, 2)} \\\\")
        if arch == "linear":
            lines.append("\\midrule")
    out["table-conditions.tex"] = "\n".join(lines) + "\n"

    # Sensitivity of the unchanged-skill difference.
    sens = read(P2A / "confirmation/descriptive-sensitivity.json")["results"]
    dirn = read(AR / "posthoc-direction.json")["effects"]
    rows = []
    def row(label, status, key):
        cells = []
        for arch in ARCHS:
            mval, lo, hi = key(arch)
            cells.append(f"{signed(mval, 3)} & {ci(lo, hi, 3)}")
        rows.append(f"{label} & {status} & " + " & ".join(cells) + " \\\\")
    pe = lambda a: s["primary"][a]["effects"]["stable"]
    row("Primary, six intervals (99.17\\%)", "prespecified", lambda a: (pe(a)["mean"], pe(a)["low"], pe(a)["high"]))
    row("Two intervals (97.5\\%)", "post hoc", lambda a: (pe(a)["mean"], posth[a]["two_way"]["low"], posth[a]["two_way"]["high"]))
    row("Family bootstrap (99.17\\%)", "descriptive", lambda a: (pe(a)["mean"], *sens[a]["stable"]["percentile_interval_99_1667"]))
    row("Initialisation 2701 only (95\\%)", "descriptive", lambda a: (sens[a]["initialization_2701"]["mean"], *sens[a]["initialization_2701"]["descriptive_95_interval"]))
    row("Initialisation 2702 only (95\\%)", "descriptive", lambda a: (sens[a]["initialization_2702"]["mean"], *sens[a]["initialization_2702"]["descriptive_95_interval"]))
    row("$f$-updates only (95\\%)", "post hoc", lambda a: (dirn[a]["unrestricted"]["f"]["mean"], dirn[a]["unrestricted"]["f"]["low"], dirn[a]["unrestricted"]["f"]["high"]))
    row("$g$-updates only (95\\%)", "post hoc", lambda a: (dirn[a]["unrestricted"]["g"]["mean"], dirn[a]["unrestricted"]["g"]["low"], dirn[a]["unrestricted"]["g"]["high"]))
    row("$f$ minus $g$ (95\\%)", "post hoc", lambda a: (dirn[a]["unrestricted"]["f_minus_g"]["mean"], dirn[a]["unrestricted"]["f_minus_g"]["low"], dirn[a]["unrestricted"]["f_minus_g"]["high"]))
    ns = lambda a: s["normalized_sensitivity"][a]["stable"]
    row("Fixed-length steps (95\\%)", "prespecified", lambda a: (ns(a)["mean"], ns(a)["low"], ns(a)["high"]))
    out["table-sensitivity.tex"] = "\n".join(rows) + "\n"

    # Families a future confirmation would need (post hoc planning arithmetic, analysis/posthoc_sample_size.py).
    ss = read(AR / "posthoc-sample-size.json")
    lines = []
    for margin in ("0.05", "0.025"):
        for k, label in ((2, "two intervals (97.5\\%)"), (6, "six intervals (99.17\\%)")):
            need = ss["families_needed"][margin][f"{k}_way"]
            lines.append(f"$\\pm${margin} & {label} & {count(need['expected_half_width_equals_margin'])} & "
                         f"{count(need['power_0_8'])} \\\\")
    out["table-samplesize.tex"] = "\n".join(lines) + "\n"

    # Study 3 table: primary and main secondary estimates, both architectures.
    r3 = read(P3 / "results/summary.json")
    def cell(x, d=3):
        return f"{signed(x['mean'], d)} {ci(x['low'], x['high'], d)}"
    def cell4(x):
        return f"{signed(1e4 * x['mean'], 2)} {ci(1e4 * x['low'], 1e4 * x['high'], 2)}"
    rows3 = [
        ("Competent pairs", lambda a: f"{r3['competence'][a]['competent_pairs']} of {r3['competence'][a]['pairs']}"),
        ("Damage, independent / joint", lambda a: f"{f(r3['primary'][a]['independent_mean'], 3)} / {f(r3['primary'][a]['joint_mean'], 3)}"),
        ("\\textbf{Difference in damage (97.5\\%)}", lambda a: "\\textbf{" + cell(r3["primary"][a]) + "}"),
        ("Decision", lambda a: r3["primary"][a]["decision"].replace("_", " ")),
        ("Changed-skill error ($\\times 10^{-4}$)", lambda a: cell4(r3["secondary"][a]["changed"])),
        ("Withheld-pair error", lambda a: cell(r3["secondary"][a]["transfer"])),
        ("Damage, $f$-updates", lambda a: cell(r3["secondary"][a]["direction"]["f"])),
        ("Damage, $g$-updates", lambda a: cell(r3["secondary"][a]["direction"]["g"])),
        ("$f$-updates minus $g$-updates", lambda a: cell(r3["secondary"][a]["direction"]["f_minus_g"])),
        ("Fixed-length steps, changed skill", lambda a: cell(r3["secondary"][a]["fixed_length"]["changed"])),
        (f"Damage, both confirmations ({r3['secondary']['linear']['pooled_with_phase2']['n']} families, 97.5\\%)",
         lambda a: cell(r3["secondary"][a]["pooled_with_phase2"])),
    ]
    out["table-study3.tex"] = "\n".join(f"{label} & " + " & ".join(fn(a) for a in ARCHS) + " \\\\" + (" \\midrule" if i == 3 else "")
                                          for i, (label, fn) in enumerate(rows3)) + "\n"

    # Phase 1 table.
    pilot = read(P1R / "pilot-analysis-v1/summary.json")
    longf = read(P1R / "long-final-analysis-v1/summary.json")
    names = {"factorwise": "Factorwise", "mixed_change": "Mixed change", "shuffled": "Shuffled"}
    lines = []
    for h, H in names.items():
        fe = [r for r in pilot["family_endpoints"] if r["optimizer_mode"] == "reset" and r["condition"] == "changed_primitive" and r["history"] == h]
        b = pilot["base_history_means"][h]
        lines.append(f"{H} & {pct(b['f'], 1)} & {pct(b['combined_seen'], 1)} & {pct(b['combined_heldout'], 1)} & "
                     f"{pct(longf['base_history_means'][h]['combined_heldout'], 1)} & "
                     f"{pct(mean([r['changed_primitive'] for r in fe]), 1)} & {pct(mean([r['unchanged_primitive'] for r in fe]), 1)} & "
                     f"{pct(mean([r['combined_heldout'] for r in fe]), 1)} \\\\")
    out["table-phase1.tex"] = "\n".join(lines) + "\n"
    return out


def render(n):
    lines = ["% Generated by paper/make_numbers.py from saved study outputs. Do not edit."]
    for k in sorted(n):
        lines.append(f"\\newcommand{{\\{k}}}{{{n[k]}}}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    # Each table ends with its closing rule: a rule after \input of a file cannot follow the last row's \\ (LaTeX
    # reads \input as a command, and \bottomrule must come directly after the row end).
    files = {"numbers.tex": render(numbers()), **{k: v + "\\bottomrule\n" for k, v in tables().items()}}
    if args.check:
        stale = [k for k, v in files.items() if not (PAPER / "generated" / k).exists() or (PAPER / "generated" / k).read_text() != v]
        extra = [p.name for p in (PAPER / "generated").glob("*.tex") if p.name not in files]
        if stale or extra:
            print("out of date: " + ", ".join(stale + extra), file=sys.stderr)
            sys.exit(1)
        return
    (PAPER / "generated").mkdir(exist_ok=True)
    for k, v in files.items():
        (PAPER / "generated" / k).write_text(v)
    print(f"wrote {len(files['numbers.tex'].splitlines()) - 1} macros and {len(files) - 1} tables")


if __name__ == "__main__":
    main()
