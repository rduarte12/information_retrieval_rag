"""
Gera as figuras do relatório a partir dos arquivos de results/.

Uso, na raiz do repositório:
    python scripts/gerar_figuras.py            # grava em results/figuras/
    python scripts/gerar_figuras.py <pasta>    # grava em outra pasta
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
R = str(ROOT) + '/'
OUT = sys.argv[1].rstrip('/') + '/' if len(sys.argv) > 1 else str(ROOT / 'results' / 'figuras') + '/'
Path(OUT).mkdir(parents=True, exist_ok=True)
INK='#191919'; INK2='#666666'; GRID='#e7e7e7'; AXIS='#c2c2c2'
BLUE,ORANGE,AQUA,YELLOW,RED='#257de0','#f47230','#00b894','#f2a600','#ec4647'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':AXIS,'axes.labelcolor':INK2,
 'xtick.color':INK2,'ytick.color':INK2,'axes.titlesize':8.5,'axes.titlecolor':INK,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6,'axes.axisbelow':True,'legend.frameon':False,'pdf.fonttype':42})
br=FuncFormatter(lambda v,_: (f'{v:.2f}'.rstrip('0').rstrip('.') if v!=0 else '0').replace('.',',').replace('-','−'))
br2=FuncFormatter(lambda v,_: f'{v:.1f}'.replace('.',',').replace('-','−'))

m=pd.read_csv(R+'results/metrics_per_query.csv').pivot(index='query_id',columns='model',values='NDCG10')

# Fig 1: sorted per-query differences, small multiples, shared y
pairs=[('semantic','lexical','Semântico − lexical'),('hybrid','lexical','Híbrido − lexical'),('reranking','hybrid','Reranking − híbrido')]
fig,axs=plt.subplots(1,3,figsize=(6.6,2.3),sharey=True)
for ax,(a,b,t) in zip(axs,pairs):
    d=(m[a]-m[b]).sort_values(ascending=False).values
    x=np.arange(len(d)); c=np.where(d>1e-12,BLUE,np.where(d<-1e-12,RED,'#efefef'))
    ax.bar(x,d,width=1.0,color=c,linewidth=0)
    ax.axhline(0,color=AXIS,lw=0.8)
    pos=(d>1e-12).sum(); neg=(d<-1e-12).sum(); tie=len(d)-pos-neg
    ax.set_title(t,loc='left',pad=12)
    ax.text(0,1.02,f'melhor {pos} · igual {tie} · pior {neg}',transform=ax.transAxes,fontsize=6.5,color=INK2,va='bottom')
    ax.set_xlim(-2,len(d)+1); ax.set_xticks([0,100,200,300]); ax.grid(axis='x',visible=False)
    ax.yaxis.set_major_formatter(br)
axs[0].set_ylabel('diferença de NDCG@10')
axs[1].set_xlabel('323 consultas do test, ordenadas pela diferença')
fig.tight_layout(w_pad=1.0); fig.savefig(OUT+'fig_diferencas.pdf'); plt.close(fig)

# Fig 2: NDCG by candidate group
c=pd.read_csv(R+'results/evidencias/indice/lexical_candidates_test.csv').set_index('query_id').n_candidates
grp=np.where(c.reindex(m.index)<100,'< 100 candidatos\n(127 consultas)','≥ 100 candidatos\n(196 consultas)')
sysn=[('lexical','Lexical',BLUE),('semantic','Semântico',ORANGE),('hybrid','Híbrido',AQUA),('reranking','Híbrido + reranking',YELLOW)]
g=m.assign(g=grp).groupby('g').mean()
fig,ax=plt.subplots(figsize=(3.25,2.5))
labels=list(g.index); x=np.arange(len(labels)); w=0.19
for i,(k,n,col) in enumerate(sysn):
    xs=x+(i-1.5)*(w+0.012); v=g[k].values
    ax.bar(xs,v,width=w,color=col,label=n,linewidth=0)
    for xi,vi in zip(xs,v): ax.text(xi,vi+0.004,f'{vi:.3f}'.replace('.',','),ha='center',va='bottom',fontsize=6,color=INK,rotation=90)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis='x',visible=False)
ax.set_ylabel('NDCG@10 médio (test)'); ax.set_ylim(0,0.47); ax.yaxis.set_major_formatter(br)
ax.legend(ncol=2,fontsize=7,loc='upper center',bbox_to_anchor=(0.5,1.22),handlelength=1,columnspacing=1)
fig.tight_layout(); fig.savefig(OUT+'fig_candidatos.pdf'); plt.close(fig)

# Fig 3: cost x quality
pts=[('Lexical',0.759,0.3281),('Semântico',8.046,0.2973),('Híbrido',0.759+8.046+0.693,0.3447),('Híbrido + reranking',0.759+8.046+0.693+2255.664,0.3266)]
fig,ax=plt.subplots(figsize=(3.25,2.5))
for n,t,v in pts:
    ax.scatter(t,v,s=36,color=BLUE,edgecolor='white',linewidth=1.5,zorder=3)
    off={'Lexical':(6,-3),'Semântico':(6,-3),'Híbrido':(6,3),'Híbrido + reranking':(-6,8)}[n]
    ha='right' if n=='Híbrido + reranking' else 'left'
    ax.annotate(f'{n}\n{v:.4f}'.replace('.',','),(t,v),xytext=off,textcoords='offset points',fontsize=7,color=INK,ha=ha,va='center')
ax.set_xscale('log'); ax.set_xlim(0.3,10000); ax.set_ylim(0.28,0.36)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'.replace('.',',')))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:.2f}'.replace('.',',')))
ax.set_xlabel('tempo acumulado por consulta (ms, escala log)'); ax.set_ylabel('NDCG@10 (test)')
fig.tight_layout(); fig.savefig(OUT+'fig_custo.pdf'); plt.close(fig)

# Fig 4a: RAG means
crit=['Fidelidade','Completude','Utilidade','Rastreab.']
A=[0.8,1.4,1.4,np.nan]; B=[2.0,1.9,1.8,1.9]
fig,ax=plt.subplots(figsize=(3.25,2.4))
x=np.arange(4); w=0.36
for xs,vals,col,n in [(x-w/2-0.01,A,BLUE,'A (sem RAG)'),(x+w/2+0.01,B,ORANGE,'B (com RAG)')]:
    for xi,vi in zip(xs,vals):
        if np.isnan(vi): ax.text(xi,0.05,'N/A',ha='center',va='bottom',fontsize=6.5,color=INK2); continue
        ax.bar(xi,vi,width=w,color=col,linewidth=0)
        ax.text(xi,vi+0.03,f'{vi:.1f}'.replace('.',','),ha='center',va='bottom',fontsize=6.5,color=INK)
ax.set_xticks(x); ax.set_xticklabels(crit); ax.grid(axis='x',visible=False)
ax.set_ylim(0,2.35); ax.set_yticks([0,1,2]); ax.set_ylabel('nota média (0 a 2)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label='A (sem RAG)'),Patch(color=ORANGE,label='B (com RAG)')],ncol=2,fontsize=7,loc='upper center',bbox_to_anchor=(0.5,1.15),handlelength=1)
fig.tight_layout(); fig.savefig(OUT+'fig_rag_medias.pdf'); plt.close(fig)

# Fig 4b: per-query delta B-A (mean F,C,U)
r=pd.read_csv(R+'results/rag_experiment.csv')
r['m']=r[['fidelidade','completude','utilidade']].mean(axis=1)
p=r.pivot(index='query_id',columns='condition',values='m'); d=(p['B']-p['A']).sort_values()
fig,ax=plt.subplots(figsize=(3.25,2.4))
y=np.arange(len(d)); cols=[BLUE if v>0 else RED for v in d.values]
ax.barh(y,d.values,height=0.7,color=cols,linewidth=0)
ax.axvline(0,color=AXIS,lw=0.8)
sel={'PLAIN-2470':'caso 1','PLAIN-153':'caso 2','PLAIN-2430':'caso 3'}
ax.set_yticks(y); ax.set_yticklabels([f'{q}  ({sel[q]})' if q in sel else q for q in d.index],fontsize=6.5)
for lab in ax.get_yticklabels():
    if lab.get_text().endswith(')'): lab.set_color(INK); lab.set_fontweight('bold')
for yi,v in zip(y,d.values):
    ax.text(v+(0.04 if v>0 else -0.04),yi,f'{v:+.2f}'.replace('.',',').replace('-','−'),va='center',ha='left' if v>0 else 'right',fontsize=6,color=INK)
ax.set_xlim(-0.75,2.05); ax.grid(axis='y',visible=False); ax.xaxis.set_major_formatter(br2)
ax.set_xlabel('B − A (média de fidelidade,\ncompletude e utilidade)')
fig.tight_layout(); fig.savefig(OUT+'fig_rag_delta.pdf'); plt.close(fig)
print(f'Figuras gravadas em {OUT}')
