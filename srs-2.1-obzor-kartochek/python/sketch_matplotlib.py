# Эскиз обзора карточек алертов (СРС 2.1)
# Запуск из корня репозитория: python python/sketch_matplotlib.py [путь_к_parquet]
# По умолчанию берёт выборку data/iavd_1000_incidents_raw_target.parquet
import sys
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
plt.rcParams['font.family']='DejaVu Sans'
PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/iavd_1000_incidents_raw_target.parquet'
DAY = '2024-05-28'                                # смена для эскиза
df = pd.read_parquet(PATH)
d=df[df['Дата'].astype(str)==DAY].copy()
d['h']=d['Создан'].dt.tz_convert('Europe/Moscow').dt.hour
d['n']=d['Количество инцидентов']
order=['Критическая','Высокая','Средняя','Низкая']
INK='#1f2328'; MUTED='#6e7781'; LINE='#d0d7de'; PLATE='#f3f5f8'; RED='#cf222e'
GR={'Критическая':RED,'Высокая':'#57606a','Средняя':'#8c959f','Низкая':'#c3c9d0'}
fig=plt.figure(figsize=(16,10),dpi=150); fig.patch.set_facecolor('white')
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,160); ax.set_ylim(0,100); ax.axis('off')
ax.text(3,96.5,'Эскиз: обзор карточек алертов · смена '+pd.Timestamp(DAY).strftime('%d.%m.%Y')+' (iavd_incidents_raw)',fontsize=17,weight='bold',color=INK,va='center')
ax.text(3,93.3,'Нейтральная база + один сигнальный цвет (Критическая). Вердикт — форма значка + подпись, не цвет.',fontsize=11,color=MUTED,va='center')
# KPI plate
ax.add_patch(FancyBboxPatch((3,82),154,8,boxstyle='round,pad=0,rounding_size=1.2',fc=PLATE,ec='none'))
nov=(d['Вердикт']=='Не указан').sum(); crit=(d['Критичность']=='Критическая'); cnv=(crit&(d['Вердикт']=='Не указан')).sum()
k=[('Карточек за смену',f'{len(d)}'),('Алертов в карточках',f'{int(d.n.sum()):,}'.replace(',',' ')),
   ('Без вердикта',f'{nov}  ({nov/len(d):.0%})'),('Критических без вердикта',f'{cnv}')]
for i,(t,v) in enumerate(k):
    x=6+i*38.5
    ax.text(x,88.2,t,fontsize=10.5,color=MUTED,va='center')
    ax.text(x,84.8,v,fontsize=19,weight='bold',color=RED if i==3 else INK,va='center')
    if i: ax.plot([x-2.5,x-2.5],[83.5,88.8],color=LINE,lw=1)
# Left: queue table
ax.text(3,78,'Очередь «Требует вердикта аналитика» — сорт.: Критичность ↓, затем число алертов ↓',fontsize=12,weight='bold',color=INK,va='center')
q=d[d['Вердикт']=='Не указан'].copy()
q['r']=q['Критичность'].map({c:i for i,c in enumerate(order)})
q=q.sort_values(['r','n'],ascending=[True,False])
rows=pd.concat([q[q.r==0].head(4),q[q.r==1].head(3),q[q.r==2].head(3),q[q.r==3].head(2)])
cols=[(4,'ID'),(13,'Время'),(22,'Критичность'),(38,'Тип инцидента'),(76,'Алертов'),(104,'Вердикт')]
y0=74
for x,t in cols: ax.text(x,y0,t,fontsize=10,color=MUTED,va='center',weight='bold')
ax.plot([3,114],[y0-1.8,y0-1.8],color=LINE,lw=1)
cap=100; maxlen=18
y=y0-4.2; prev=None
for _,r in rows.iterrows():
    c=r['Критичность']
    if prev is not None and c!=prev: y-=1.4
    prev=c
    if c=='Критическая':
        ax.add_patch(Rectangle((3,y-1.7),111,3.4,fc='#ffebe9',ec='none'))
    ax.add_patch(Rectangle((3,y-1.7),0.8,3.4,fc=GR[c],ec='none'))
    tc=RED if c=='Критическая' else INK
    ax.text(4.5,y,str(r.EntityId),fontsize=10,color=INK,va='center',family='DejaVu Sans Mono')
    ax.text(13,y,r['Создан'].tz_convert('Europe/Moscow').strftime('%H:%M'),fontsize=10,color=INK,va='center')
    ax.text(22,y,c,fontsize=10,color=tc,va='center',weight='bold' if c=='Критическая' else 'normal')
    t=r['Тип инцидента']; t=t if len(t)<=34 else t[:33]+'…'
    ax.text(38,y,t,fontsize=10,color=INK,va='center')
    n=0 if pd.isna(r.n) else r.n; L=min(n,cap)/cap*maxlen
    ax.add_patch(Rectangle((76,y-0.9),max(L,0.2),1.8,fc='#57606a',ec='none'))
    if n>cap: ax.text(76+L+0.3,y,'»',fontsize=10,color=INK,va='center',weight='bold')
    ax.text(76+L+(2 if n>cap else 0.8),y,f'{int(n)}',fontsize=9.5,color=INK,va='center')
    ax.add_patch(Circle((105,y),0.85,fc='white',ec=INK,lw=1.6))
    ax.text(107,y,'нет',fontsize=9.5,color=INK,va='center')
    y-=3.9
ax.text(3,y+0.6,f'… ещё {len(q)-len(rows)} карточек в очереди. Длина = число алертов на общей нулевой линии; ≥{cap} — обрезано, «»».',fontsize=9,color=MUTED,va='center')
# Right top: criticality x verdict bars
X0=122; W=34
ax.text(X0-2,78,'Критичность: с вердиктом / без',fontsize=12,weight='bold',color=INK,va='center')
g=d.groupby('Критичность')['Вердикт'].apply(lambda s:( (s!='Не указан').sum(), (s=='Не указан').sum()))
mx=max(a+b for a,b in g)
yy=73
for c in order:
    a,b=g.get(c,(0,0))
    ax.text(X0-2,yy,c,fontsize=10,color=RED if c=='Критическая' else INK,va='center',ha='left')
    bx=X0+11; s=W-4
    la=a/mx*(s-10); lb=b/mx*(s-10)
    ax.add_patch(Rectangle((bx,yy-1.1),la,2.2,fc='#57606a',ec='none'))
    ax.add_patch(Rectangle((bx+la+0.25,yy-1.1),lb,2.2,fc='white',ec='#57606a',lw=1.2,hatch='////'))
    ax.text(bx+la+lb+1,yy,f'{a} / {b}',fontsize=9.5,color=INK,va='center')
    yy-=4.2
ax.plot([X0+11,X0+11],[yy+2,74.8],color=INK,lw=1)
ax.add_patch(Rectangle((X0-2,yy-0.3),2,1.4,fc='#57606a')); ax.text(X0+0.8,yy+0.4,'с вердиктом',fontsize=9,color=MUTED,va='center')
ax.add_patch(Rectangle((X0+15,yy-0.3),2,1.4,fc='white',ec='#57606a',hatch='////')); ax.text(X0+17.8,yy+0.4,'без вердикта',fontsize=9,color=MUTED,va='center')
# Right mid: verdict legend
vy=yy-6
ax.text(X0-2,vy,'Кодировка Вердикта: форма + подпись',fontsize=12,weight='bold',color=INK,va='center')
vc=d['Вердикт'].value_counts()
items=[('Не указан → «нужен вердикт»','o_empty'),('False Positive','o_full'),('Дубликат','o_dash'),('Решён','o_check')]
vy-=4
for lab,kind in items:
    cx=X0
    if kind=='o_empty': ax.add_patch(Circle((cx,vy),0.85,fc='white',ec=INK,lw=1.6))
    if kind=='o_full': ax.add_patch(Circle((cx,vy),0.85,fc='#8c959f',ec='#8c959f'))
    if kind=='o_dash': ax.add_patch(Circle((cx,vy),0.85,fc='white',ec='#8c959f',lw=1.2)); ax.plot([cx-0.6,cx+0.6],[vy-0.6,vy+0.6],color='#8c959f',lw=1.2)
    if kind=='o_check': ax.add_patch(Circle((cx,vy),0.85,fc='#8c959f',ec='#8c959f')); ax.text(cx,vy,'✓',color='white',fontsize=8,ha='center',va='center',weight='bold')
    key=lab.split(' →')[0]
    ax.text(cx+2,vy,f'{lab}   ({vc.get(key,0)})',fontsize=10,color=INK,va='center')
    vy-=3.3
# Bottom: hourly timeline
ax.text(3,17,'Динамика: карточки по часам смены (МСК)',fontsize=12,weight='bold',color=INK,va='center')
hh=d.groupby('h').size().reindex(range(7,18),fill_value=0)
hc=d[crit].groupby('h').size().reindex(range(7,18),fill_value=0)
xs=np.linspace(8,150,len(hh)); base=3; sc=10/hh.max()
ax.plot([6,152],[base,base],color=LINE,lw=1)
ax.plot(xs,base+hh.values*sc,color='#57606a',lw=2,marker='o',ms=5)
for x,h,v,cv in zip(xs,hh.index,hh.values,hc.values):
    ax.text(x,base-1.6,f'{h}:00',fontsize=9,color=MUTED,ha='center',va='center')
im=hh.values.argmax()
ax.text(xs[im]+2,base+hh.values[im]*sc,f'пик {hh.values[im]} карточек, {hh.index[im]}:00',fontsize=9.5,color=INK,va='center')
ax.text(152,15,'Нейтральная линия: сигнальный цвет не тратится на обзорный график',fontsize=9,color=MUTED,ha='right',va='center')
fig.savefig('images/sketch_1000.png',dpi=150,facecolor='white')
print('Сохранено: images/sketch_1000.png')
