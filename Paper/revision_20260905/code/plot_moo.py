"""Plot the locked multi-method experiment and scenario interpretation."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;OUT=ROOT/"figures";R=ROOT/"results"
report=json.loads((R/"analysis_summary.json").read_text(encoding="utf-8"))
primary=report["primary_method"]
METHODS=["NSGA2","SPEA2","MOEAD","UniformRandom","LatinHypercube"]
LABELS={"NSGA2":"NSGA-II","SPEA2":"SPEA2","MOEAD":"MOEA/D","UniformRandom":"Uniform random","LatinHypercube":"Latin hypercube"}
COLORS=dict(zip(METHODS,["#0072B2","#D55E00","#009E73","#777777","#CC79A7"]))
MARKERS=dict(zip(METHODS,["o","s","^","x","D"]))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,"axes.labelsize":10,
                    "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"axes.spines.top":False,
                    "axes.spines.right":False,"savefig.dpi":400,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
metadata=[]
def save(fig,name,caption,sources):
    fig.canvas.draw()
    for ext in ["png","pdf","svg"]:fig.savefig(OUT/f"{name}.{ext}",facecolor="white")
    metadata.append(dict(id=name,caption=caption,width_in=float(fig.get_figwidth()),height_in=float(fig.get_figheight()),dpi=400,sources={str(s.relative_to(PAPER)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}))
    plt.close(fig)

seed=pd.read_csv(R/"evaluation_seed_means.csv");history=pd.read_csv(R/"convergence.csv")
fig,axes=plt.subplots(2,2,figsize=(7.1,6.1),layout="constrained")
for panel,metric,label in [(0,"hv","Hypervolume"),(1,"igd_plus","IGD+"),(3,"seconds","Seconds per scenario")]:
    ax=axes.flat[panel]
    for i,method in enumerate(METHODS):
        values=seed[seed.method==method][metric]
        ax.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt=MARKERS[method],color=COLORS[method],markersize=5,capsize=3)
        ax.scatter(values,np.full(len(values),i)+np.linspace(-.09,.09,len(values)),s=8,color=COLORS[method],alpha=.5)
    ax.set_yticks(range(5),[LABELS[m] for m in METHODS]);ax.invert_yaxis();ax.set_xlabel(label);ax.set_title(f"({chr(97+panel)})",loc="left",fontweight="bold")
    if metric!="hv":ax.set_xlim(left=0)
ax=axes[1,0]
for method in METHODS:
    tab=history[history.method==method].groupby("n_eval").hv_attainment.mean()
    ax.plot(tab.index,tab.values,color=COLORS[method],marker=MARKERS[method],markersize=3,lw=1.2,label=LABELS[method])
ax.set(xlabel="Proxy evaluations per run",ylabel="HV coverage (%)");ax.set_ylim(min(95,history.groupby(["method","n_eval"]).hv_attainment.mean().min()-.3),100.1);ax.set_title("(c)",loc="left",fontweight="bold")
ax.legend(loc="lower left",frameon=False)
save(fig,"Fig_5","图5 固定参数后的多方法评价。a为HV，b为IGD+，c为外部档案随调用数变化的经验参考前沿HV覆盖率，d为分摊计算时间。a、b、d中小点为5个种子各自在22个情景上的均值，大点及横线为种子均值±SD；c为全部情景与种子的描述性均值，纵轴为覆盖率局部范围。",[R/"evaluation_seed_means.csv",R/"convergence.csv"])

points=pd.read_csv(R/"pareto_points.csv");solutions=pd.read_csv(R/"representative_solutions.csv")
fig,axes=plt.subplots(2,2,figsize=(7.1,6.2),layout="constrained")
for panel,(ax,date) in enumerate(zip(axes.flat,report["representative_dates"])):
    for method in METHODS:
        tab=points[(points.date==date)&(points.method==method)]
        # Display all archived points; no interpolation between evaluated candidates.
        ax.scatter(tab.TN_out,tab.DEC,color=COLORS[method],marker=MARKERS[method],s=11,alpha=.45,label=LABELS[method],linewidths=.6)
    sol=solutions[(solutions.date==date)&(solutions.role=="compromise")].iloc[0]
    ax.scatter([sol.TN_out],[sol.DEC],s=70,c="black",marker="*",label="Selected compromise",zorder=5)
    ax.scatter([sol.baseline_TN],[sol.baseline_DEC],s=35,c="black",marker="+",label="Historical proxy",zorder=5)
    ax.set(xlabel="Predicted TN_out (mg/L)",ylabel="Predicted DEC (kWh/d)");ax.set_title(f"({chr(97+panel)}) {date}",loc="left",fontweight="bold")
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
save(fig,"Fig_6","图6 四个代表情景的双目标解集。日期按7、9、11、12月中旬邻近的已选情景确定，未按收益筛选。彩色点保留各方法5个种子的非支配档案，黑色星号为主要方法合并档案的折衷解，加号为同背景历史输入代理值。面板使用各自的物理坐标范围；点间没有补造或插值结果。",[R/"pareto_points.csv",R/"representative_solutions.csv"])

matched=pd.read_csv(R/"solutions_with_feasible_anchor.csv")
comp=matched[matched.role=="compromise"].sort_values("date").copy();comp["delta_PPA"]=comp.PPA-comp.historical_PPA;comp["delta_DO"]=comp.DO-comp.historical_DO
fig,axes=plt.subplots(2,2,figsize=(7.1,5.4),layout="constrained")
for i,(column,label) in enumerate([("delta_TN","TN_out change (mg/L)"),("delta_DEC","DEC change (kWh/d)"),("delta_PPA","PPA coordinate change (kg)"),("delta_DO","DO coordinate change (mg/L)")]):
    ax=axes.flat[i];dt=pd.to_datetime(comp.date);supported=comp.historical_in_search_domain
    ax.scatter(dt[supported],comp.loc[supported,column],s=24,color=COLORS[primary],marker="o",label="Historical point in domain")
    ax.scatter(dt[~supported],comp.loc[~supported,column],s=30,color="#D55E00",marker="^",label="Historical point outside domain")
    if i<2:
        anchor_column="delta_anchor_TN" if i==0 else "delta_anchor_DEC"
        ax.scatter(dt,comp[anchor_column],s=27,facecolors="none",edgecolors="#333333",marker="s",label="Relative to feasible anchor")
    ax.axhline(0,ls="--",lw=.8,color="#666666");ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"));ax.set(xlabel="2025 scenario date",ylabel=label);ax.set_title(f"({chr(97+i)})",loc="left",fontweight="bold")
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=1,frameon=False)
save(fig,"Fig_7","图7 主要方法折衷方案的情景变化。a、b中实心点为相对历史输入的代理差值，圆点和三角区分历史点是否在搜索区间内；空心方块为相对同域可行参照的差值。c、d为相对历史输入的平滑坐标变化。可行参照由历史点逐坐标投影到预定区间得到，未参与算法调优；未连接不连续日期。",[R/"solutions_with_feasible_anchor.csv"])

unc=pd.read_csv(R/"solution_anchor_residual_sensitivity.csv");unc=unc[unc.correlation==.5].sort_values("date")
fig,axes=plt.subplots(1,2,figsize=(7.1,6.8),layout="constrained",sharey=True)
for j,(target,column) in enumerate([("TN","delta_anchor_TN"),("DEC","delta_anchor_DEC")]):
    center=comp.set_index("date").loc[unc.date,column].to_numpy();lo=unc[f"{target}_q025"].to_numpy();hi=unc[f"{target}_q975"].to_numpy();y=np.arange(len(unc));ax=axes[j]
    ax.errorbar(center,y,xerr=np.vstack([center-lo,hi-center]),fmt="o",markersize=3.5,color=COLORS[primary],capsize=2,lw=.8)
    ax.axvline(0,color="#333333",ls="--",lw=.9);ax.set_yticks(y,unc.date);ax.set_xlabel("TN_out change (mg/L)" if j==0 else "DEC change (kWh/d)");ax.set_title(f"({chr(97+j)})",loc="left",fontweight="bold")
axes[0].invert_yaxis()
save(fig,"Fig_S5","图S5 折衷方案相对同域可行参照差值的经验残差扰动。点为模型预测差值，横线为相关系数假设0.5下20000次联合残差扰动的2.5%—97.5%分位。该范围显示差值与外层误差尺度的关系，不代表经过现场反事实数据校准的置信区间。",[R/"solutions_with_feasible_anchor.csv",R/"solution_anchor_residual_sensitivity.csv"])

fixed=pd.read_csv(R/"fixed_do_comparison.csv");scores=pd.read_csv(R/"evaluation_metrics.csv")
fig,axes=plt.subplots(1,2,figsize=(7.1,3.4),layout="constrained")
for state,marker,color in [(True,"o",COLORS[primary]),(False,"^","#D55E00")]:
    tab=fixed[fixed.fixed_do_historical_supported==state]
    axes[0].scatter(tab.joint_minus_fixed_TN,tab.joint_minus_fixed_DEC,s=24,marker=marker,color=color,label="Historical DO" if state else "Boundary DO")
axes[0].axhline(0,color="#888888",lw=.7);axes[0].axvline(0,color="#888888",lw=.7);axes[0].set(xlabel="Joint minus fixed DO: TN (mg/L)",ylabel="Joint minus fixed DO: DEC (kWh/d)");axes[0].legend(frameon=False);axes[0].set_title("(a)",loc="left",fontweight="bold")
for method in METHODS[:3]:
    effects=[]
    for reference,column in [(1.05,"hv_ref_1.05"),(1.1,"hv"),(1.2,"hv_ref_1.2")]:
        piv=scores[scores.method.isin([method,"UniformRandom"])].pivot(index=["date","seed"],columns="method",values=column)
        effects.append(float((piv[method]-piv.UniformRandom).mean()))
    axes[1].plot([1.05,1.1,1.2],effects,marker=MARKERS[method],color=COLORS[method],label=LABELS[method])
axes[1].set(xlabel="HV reference coordinate",ylabel="Mean HV difference vs random");axes[1].set_xticks([1.05,1.1,1.2]);axes[1].legend(frameon=False);axes[1].set_title("(b)",loc="left",fontweight="bold")
save(fig,"Fig_S6","图S6 固定DO与HV参考点敏感性。a为联合搜索折衷方案减固定DO搜索折衷方案的双目标差值，圆点固定历史DO，三角表示历史DO超出搜索区间而固定于最近边界；b为三个HV参考点下各演化方法相对均匀随机的配对平均HV差。",[R/"fixed_do_comparison.csv",R/"evaluation_metrics.csv"])
(OUT/"moo_figure_manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"figures":len(metadata),"primary":primary}))
