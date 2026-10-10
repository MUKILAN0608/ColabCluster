"""Read-only, seeded round-bootstrap analysis and standalone benchmark figures."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics as stats
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.validate_scaling import METHOD, FIELDS
from common.scaling import VALIDATION_SIZES, WORKER_IDS, partition_samples

NUMERIC = ("client_wall_time_ms","server_wall_time_ms","gpu_inference_time_ms","effective_throughput_images_per_second")


def percentile(values, q):
    ordered=sorted(values)
    index=(len(ordered)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def interval(values):
    return [percentile(values,.025),percentile(values,.975)] if values else None


def compatible_metadata(left,right):
    """No silent pooling of legacy, changed-code or changed-runtime experiments."""
    return all(left.get(k)==right.get(k) for k in ("method","seed","workers")) and left.get("method")==METHOD and (
        left.get("software",{}).get("source_sha256")==right.get("software",{}).get("source_sha256"))


def load(directory):
    directory=Path(directory)
    metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    if metadata.get("method")!=METHOD:
        raise ValueError("Incompatible/legacy experiment methodology; do not pool or infer round pairing")
    if set(metadata.get("targets",{})) != set(map(str,VALIDATION_SIZES)):
        raise ValueError("Missing workload targets")
    for n in VALIDATION_SIZES:
        if metadata["targets"][str(n)] < (20 if n in (1024,4096) else 5):
            raise ValueError("Trial target below experimental minimum")
    with (directory/"measurements.csv").open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):raise ValueError("Missing measurement columns")
        raw=list(reader)
    rows=[];issues=[];seen=set();versions={}
    for row in raw:
        item=dict(row)
        try:
            for field in ("run","round","order","total_samples","worker_count","seed"):
                item[field]=int(row[field])
            key=(item["total_samples"],item["round"],item["worker_count"])
            if key in seen:raise ValueError("Duplicate round/configuration")
            seen.add(key)
            if item["experiment_id"]!=metadata["experiment_id"] or item["seed"]!=metadata["seed"]:
                raise ValueError("Experiment identity/seed mismatch")
            n,c=item["total_samples"],item["worker_count"]
            if n not in VALIDATION_SIZES or c not in (1,2) or item["order"] not in (1,2) or item["round"]<1:
                raise ValueError("Invalid workload/round/order")
            if row["status"]=="passed":
                for field in NUMERIC:
                    item[field]=float(row[field])
                    if not math.isfinite(item[field]) or item[field]<=0:raise ValueError(f"Invalid {field}")
                counts=partition_samples(n,c)
                if (json.loads(row["worker_ids"])!=list(WORKER_IDS[:c]) or json.loads(row["gpu_models"])!=["Tesla T4"]*c
                    or json.loads(row["assigned_samples"])!=counts or json.loads(row["output_shapes"])!=[[v,10] for v in counts]
                    or row["http_status"]!="200" or row["api_status"]!="passed"):
                    raise ValueError("Invalid GPU identity/status/output workload")
                recorded=json.loads(row["worker_versions"])
                if set(recorded)!=set(WORKER_IDS[:c]):raise ValueError("Missing observed worker software")
                for name,version in recorded.items():
                    if not version.get("cuda_version") or not version.get("torch_version"):raise ValueError("Missing CUDA/PyTorch version")
                    if name in versions and versions[name]!=version:raise ValueError("Changed runtime software")
                    versions[name]=version
            elif row["status"]!="failed":raise ValueError("Unrecognized status")
        except (ValueError,TypeError,KeyError) as exc:
            # Invalid metadata cannot be assigned to a group reliably; fail closed.
            raise ValueError(f"Invalid measurement run {row.get('run')}: {exc}") from exc
        rows.append(item)
    metadata["observed_versions"]=versions
    return metadata,rows,issues


def mean_ci(rows,seed,resamples):
    if len(rows)<5:return None
    rng=random.Random(seed)
    # One observation per configuration/round; bootstrap its observed rounds.
    strata={}
    for r in rows:strata.setdefault(r["order"],[]).append(r["client_wall_time_ms"])
    values=[]
    for _ in range(resamples):
        sample=[v for group in strata.values() for v in rng.choices(group,k=len(group))]
        values.append(stats.mean(sample))
    return interval(values)


def group_summary(rows,size,count,target,seed,resamples):
    attempted=[r for r in rows if r["total_samples"]==size and r["worker_count"]==count]
    good=[r for r in attempted if r["status"]=="passed"]
    wall=[r["client_wall_time_ms"] for r in good];server=[r["server_wall_time_ms"] for r in good]
    def mean(values):return stats.mean(values) if values else None
    ci=mean_ci(good,seed,resamples)
    return dict(samples=size,workers=count,target=target,successful=len(good),failed=len(attempted)-len(good),complete=len(good)>=target,
        mean_client_ms=mean(wall),median_client_ms=stats.median(wall) if wall else None,
        sample_sd_ms=stats.stdev(wall) if len(wall)>1 else None,min_client_ms=min(wall) if wall else None,
        max_client_ms=max(wall) if wall else None,iqr_ms=percentile(wall,.75)-percentile(wall,.25) if wall else None,
        mean_server_ms=mean(server),median_server_ms=stats.median(server) if server else None,
        mean_summed_gpu_ms=mean([r["gpu_inference_time_ms"] for r in good]),
        mean_reported_throughput=mean([r["effective_throughput_images_per_second"] for r in good]),
        mean_ci_low=ci[0] if ci else None,mean_ci_high=ci[1] if ci else None)


def comparison(rows,a,b,seed,resamples):
    out=dict(samples=a["samples"],complete=a["complete"] and b["complete"],speedup=None,
             wall_change_percent=None,throughput_change_percent=None,difference_ms=None,
             speedup_ci=None,difference_ci=None,paired_rounds=0,unpaired_rounds=0,ci_method="unavailable")
    if not out["complete"]:return out
    out.update(speedup=a["mean_client_ms"]/b["mean_client_ms"],
        difference_ms=b["mean_client_ms"]-a["mean_client_ms"],
        wall_change_percent=(b["mean_client_ms"]/a["mean_client_ms"]-1)*100,
        throughput_change_percent=(b["mean_reported_throughput"]/a["mean_reported_throughput"]-1)*100)
    rounds={}
    for r in rows:
        if r["total_samples"]==a["samples"] and r["status"]=="passed":rounds.setdefault(r["round"],[]).append(r)
    strata={}
    for group in rounds.values():
        first=group[0]["worker_count"] if group[0]["order"]==1 else 3-group[0]["worker_count"]
        strata.setdefault(first,[]).append(group)
    out["paired_rounds"]=sum(len(group)==2 for group in rounds.values())
    out["unpaired_rounds"]=sum(len(group)==1 for group in rounds.values())
    if out["paired_rounds"]<5:return out
    # Resample whole rounds jointly, stratified by planned first configuration.
    rng=random.Random(seed);ratios=[];differences=[]
    for _ in range(resamples):
        selected=[r for groups in strata.values() for group in rng.choices(groups,k=len(groups)) for r in group]
        one=[r["client_wall_time_ms"] for r in selected if r["worker_count"]==1]
        two=[r["client_wall_time_ms"] for r in selected if r["worker_count"]==2]
        if one and two:
            x,y=stats.mean(one),stats.mean(two);ratios.append(x/y);differences.append(y-x)
    out.update(speedup_ci=interval(ratios),difference_ci=interval(differences),
               ci_method="95% percentile round-cluster bootstrap, stratified by first configuration")
    return out


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plots(output,rows,groups,comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    def save(fig,name):
        if name != "trials":
            for axis in fig.axes:
                axis.set_xticks(VALIDATION_SIZES)
                axis.tick_params(axis="x", labelrotation=35)
                axis.set_xlim(128,4224)
                if not rows:
                    axis.set_yticks([])
                elif name != "speedup":
                    axis.set_ylim(bottom=0)
        fig.tight_layout();fig.savefig(output/(name+".png"),dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for count in (1,2):
        points=[g for g in groups if g["workers"]==count and g["successful"]]
        xs=[g["samples"] for g in points];ys=[g["mean_client_ms"] for g in points]
        ax.plot(xs,ys,"o-",label=f"{count} worker(s)")
        for g in points:
            if g["mean_ci_low"] is not None:ax.vlines(g["samples"],g["mean_ci_low"],g["mean_ci_high"],color="black",alpha=.55)
    ax.set(xlabel="Workload (samples)",ylabel="Mean client wall time (ms)",title="Client wall time; bars: 95% percentile bootstrap CI")
    if not rows:ax.text(.5,.5,"No live measurements; experiment incomplete",ha="center",transform=ax.transAxes)
    ax.legend();save(fig,"wall_time")
    fig,ax=plt.subplots(figsize=(9,5));points=[c for c in comparisons if c["complete"]]
    ax.axhline(1,color="gray",linestyle="--",label="1.0x (equal means)")
    ax.plot([c["samples"] for c in points],[c["speedup"] for c in points],"o-",label="Single / two-worker mean")
    for c in points:
        if c["speedup_ci"]:ax.vlines(c["samples"],*c["speedup_ci"],color="black")
    ax.set(xlabel="Workload (samples)",ylabel="Speedup (ratio)",title="Completed configurations only; bars: 95% round-bootstrap CI")
    if not points:ax.text(.5,.5,"No completed comparison",ha="center",transform=ax.transAxes)
    ax.legend();save(fig,"speedup")
    fig,ax=plt.subplots(figsize=(9,5))
    for count in (1,2):
        points=[g for g in groups if g["workers"]==count and g["successful"]]
        ax.plot([g["samples"] for g in points],[g["mean_reported_throughput"] for g in points],"o-",label=f"{count} worker(s)")
    ax.set(xlabel="Workload (samples)",ylabel="Mean reported throughput (images/s; server wall time)",title="Reported throughput; descriptive means, no CI")
    if not rows:ax.text(.5,.5,"No live measurements",ha="center",transform=ax.transAxes)
    ax.legend();save(fig,"throughput")
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for size,ax in zip(VALIDATION_SIZES,axes.flat):
        for count in (1,2):
            good=[r for r in rows if r["total_samples"]==size and r["worker_count"]==count and r["status"]=="passed"]
            ax.scatter([r["round"] for r in good],[r["client_wall_time_ms"] for r in good],label=f"{count} worker(s)",s=20)
        ax.set(title=f"{size} samples",xlabel="Experimental round",ylabel="Client wall time (ms)");ax.legend(fontsize=7)
    fig.suptitle("Every successful observation retained; failures listed in report",y=1.01)
    save(fig,"trials")


def analyze(directory, seed=None, resamples=5000):
    if resamples<100:raise ValueError("Use at least 100 bootstrap resamples (default 5000)")
    directory=Path(directory);metadata,rows,issues=load(directory)
    seed=metadata["seed"] if seed is None else seed
    output=directory/"analysis";output.mkdir(exist_ok=True)
    groups=[group_summary(rows,n,c,metadata["targets"][str(n)],seed+n+c,resamples) for n in VALIDATION_SIZES for c in (1,2)]
    comparisons=[comparison(rows,groups[i],groups[i+1],seed+groups[i]["samples"],resamples) for i in range(0,len(groups),2)]
    write_csv(output/"summary.csv",groups);write_csv(output/"comparisons.csv",comparisons)
    available=[c["samples"] for c in comparisons if c["complete"] and c["speedup"]>1]
    smallest=min(available) if available else None
    confirm=next(c for c in comparisons if c["samples"]==4096)
    supported=bool(confirm["complete"] and confirm["difference_ci"] and confirm["difference_ci"][1]<0)
    source_hashes={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ("measurements.csv","metadata.json")}
    analysis=dict(seed=seed,resamples=resamples,confidence=.95,groups=groups,comparisons=comparisons,
        smallest_observed_advantage=smallest,advantage_4096_supported=supported,source_sha256=source_hashes,
        analysis_software=dict(python=sys.version.split()[0],matplotlib=importlib.metadata.version("matplotlib"),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (output/"analysis.json").write_text(json.dumps(analysis,indent=2,allow_nan=False),encoding="utf-8")
    plots(output,rows,groups,comparisons)
    f=lambda x:"N/A" if x is None else f"{x:.3f}"
    lines=["# ColabCluster repeatability and crossover validation", "", "## Setup and provenance", "",
        f"Experiment: `{metadata['experiment_id']}`. Raw attempts: {len(rows)}. Inputs are read-only; SHA256 hashes are in analysis.json.",
        f"Stop reason: {metadata.get('stop_reason') or 'None recorded'}.",
        f"Software: {json.dumps(metadata.get('software',{}),sort_keys=True)}", "",
        "Recorded workers (no tunnel addresses or credentials):", "```json",json.dumps(metadata.get("workers",{}),indent=2),"```",
        "Observed response CUDA/PyTorch versions:","```json",json.dumps(metadata.get("observed_versions",{}),indent=2),"```",
        "Model/input configuration:","```json",json.dumps(metadata["method"],indent=2),"```", "",
        "## Descriptive measurements", "",
        "| Samples | Workers | Passed/target | Failed | Mean ms | Median ms | SD ms | IQR ms | Min/max ms | Mean CI ms | Mean/median server ms | Summed GPU ms | Reported images/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['samples']} | {g['workers']} | {g['successful']}/{g['target']} | {g['failed']} | {f(g['mean_client_ms'])} | {f(g['median_client_ms'])} | {f(g['sample_sd_ms'])} | {f(g['iqr_ms'])} | {f(g['min_client_ms'])}/{f(g['max_client_ms'])} | {f(g['mean_ci_low'])}, {f(g['mean_ci_high'])} | {f(g['mean_server_ms'])}/{f(g['median_server_ms'])} | {f(g['mean_summed_gpu_ms'])} | {f(g['mean_reported_throughput'])} |")
    lines += ["", "## Calculated effects (only when both targets are met)", "",
        "| Samples | Complete | Speedup | Wall change % | Difference two-minus-one ms | Throughput change % | Speedup CI | Difference CI ms | Paired/unpaired rounds |",
        "|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        lines.append(f"| {c['samples']} | {c['complete']} | {f(c['speedup'])} | {f(c['wall_change_percent'])} | {f(c['difference_ms'])} | {f(c['throughput_change_percent'])} | {c['speedup_ci'] or 'N/A'} | {c['difference_ci'] or 'N/A'} | {c['paired_rounds']}/{c['unpaired_rounds']} |")
    lines += ["", "## Interpretation", "",
        f"4096-sample repeatable advantage supported under the stated bootstrap assumptions: **{'yes' if supported else 'not established'}**.",
        f"Smallest tested workload with a completed measured mean advantage: **{smallest if smallest else 'not established'}**. Neighboring tested workloads and uncertainty are shown above; this is not an exact crossover threshold.",
        "A lower mean alone is not called statistically significant. Confirmation requires the full target and a two-minus-one 95% interval wholly below zero. This is conditional evidence within this session, not proof of universal scaling or persistence across Colab sessions.",
        "If confirmation is absent or uncertain, the next experiment is a fresh fully updated session with the same balanced design and more independent rounds/sessions, not a scheduler or architectural change.", "",
        "## Statistical method and limitations", "",
        f"Analysis seed {seed}; {resamples} resamples; 95% percentile intervals. Means use order-stratified resampling of observed rounds. Comparisons resample whole rounds jointly within first-configuration strata, retaining one/two-worker dependence and unpaired successful observations. Fewer than five paired rounds suppress comparison intervals. Missing/failing trials are counted; successful outliers are never removed.",
        "Assumptions: rounds are exchangeable within order strata; both configurations in a round share comparable conditions. Time drift/autocorrelation across rounds, failures not missing at random and changes across workloads can invalidate nominal coverage. With five exploratory rounds intervals are particularly unstable. Pointwise intervals across eight workloads are not multiplicity-adjusted; crossover selection is exploratory, not a confirmatory significance claim.",
        "Client perf_counter covers complete HTTP response receipt before parsing. Server wall time includes remote model/input setup, warmup, dispatch and validation. Summed GPU time is compute across devices, never parallel elapsed time. Reported throughput is the mean of server-based sample rates, not samples divided by mean client wall time.",
        "The two confirmation sizes run first. Exploratory sizes use the same session/method and confirmation observations are reused, not rerun or merged with older data. Older five-trial experiments lack sufficient round/seed/source metadata and are deliberately not pooled. Workload order and preflight traffic remain possible confounders. Worker/model inputs match but a random untrained model and synthetic data do not measure application accuracy.",
        "Bootstrap reference: [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). The local implementation uses the standard library and round-level sampling rather than adding SciPy.", "",
        "## Failed attempts / incomplete configurations", ""]
    failures=[r for r in rows if r["status"]!="passed"]
    lines += [f"- Run {r['run']}, samples {r['total_samples']}, workers {r['worker_count']}, HTTP {r['http_status']}: {r['error']}" for r in failures] or ["No failed inference attempts recorded. Zero attempts after a blocked preflight is not a successful experiment."]
    lines += [f"- Incomplete: {g['samples']} samples, {g['workers']} worker(s): {g['successful']}/{g['target']} successful." for g in groups if not g["complete"]]
    lines += ["", "## Figures", "", "![Client wall time](wall_time.png)","![Speedup](speedup.png)","![Reported throughput](throughput.png)","![Trial variability](trials.png)"]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return analysis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--resamples",type=int,default=5000)
    args=parser.parse_args();result=analyze(args.directory,args.seed,args.resamples)
    print(f"Report: {args.directory/'analysis/report.md'}")
    print(f"4096 advantage supported: {result['advantage_4096_supported']}")


if __name__=="__main__":main()
