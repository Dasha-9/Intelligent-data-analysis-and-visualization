# Рисует траекторию взгляда (F-паттерн) поверх скриншотов.
# Запуск из корня репозитория: python python/draw_gaze.py
# Исходные скриншоты: images/raw/, результат: images/*_F.png
import math
from PIL import Image, ImageDraw, ImageFont
import os
B=next((p for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','C:/Windows/Fonts/arialbd.ttf','/Library/Fonts/Arial Bold.ttf'] if os.path.exists(p)),'DejaVuSans-Bold.ttf')
ORG=(242,120,20); WHITE=(255,255,255); INK=(31,35,40)
def arrow(d,p,q,w=9,col=ORG,dash=False):
    x1,y1=p;x2,y2=q;L=math.hypot(x2-x1,y2-y1);ux,uy=(x2-x1)/L,(y2-y1)/L
    hx,hy=x2-ux*34,y2-uy*34
    if dash:
        s=0
        while s<L-40:
            e=min(s+28,L-40); d.line([(x1+ux*s,y1+uy*s),(x1+ux*e,y1+uy*e)],fill=col,width=w); s+=46
    else:
        d.line([(x1,y1),(hx,hy)],fill=WHITE,width=w+6); d.line([(x1,y1),(hx,hy)],fill=col,width=w)
    px,py=-uy,ux
    tri=[(x2,y2),(hx+px*17,hy+py*17),(hx-px*17,hy-py*17)]
    d.polygon(tri,fill=col,outline=WHITE)
def badge(d,c,n,r=30,col=ORG):
    x,y=c; d.ellipse([x-r-4,y-r-4,x+r+4,y+r+4],fill=WHITE); d.ellipse([x-r,y-r,x+r,y+r],fill=col)
    f=ImageFont.truetype(B,32); d.text((x,y),str(n),font=f,fill=WHITE,anchor='mm')
def note(d,xy,text,size=24,col=ORG,anchor='la'):
    f=ImageFont.truetype(B,size); bb=d.textbbox(xy,text,font=f,anchor=anchor)
    d.rounded_rectangle([bb[0]-10,bb[1]-7,bb[2]+10,bb[3]+7],radius=8,fill=(255,255,255,235),outline=col,width=3)
    d.text(xy,text,font=f,fill=INK,anchor=anchor)
def zone(d,box,col=(207,34,46),w=6):
    x1,y1,x2,y2=box; s=0
    for (a,b) in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
        L=math.hypot(b[0]-a[0],b[1]-a[1]);ux,uy=(b[0]-a[0])/L,(b[1]-a[1])/L;s=0
        while s<L: e=min(s+24,L); d.line([(a[0]+ux*s,a[1]+uy*s),(a[0]+ux*e,a[1]+uy*e)],fill=col,width=w); s+=40

# ---------- Qlik ----------
im=Image.open('images/raw/qlik_do.png').convert('RGBA'); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
Y=306
arrow(d,(52,Y),(1010,Y)); arrow(d,(1090,Y),(1890,Y))          # полоса 1: KPI, якорь – красное 408
arrow(d,(52,437),(900,437))                                  # полоса 2: короче, по таблице
arrow(d,(14,470),(14,1080))                                  # вниз по левому краю к легенде
badge(d,(52,Y),1); badge(d,(1050,Y),2); badge(d,(52,437),3); badge(d,(14,1110),4,r=26)
note(d,(1000,345),'якорь: единственный цвет',20,anchor='ra')
zone(d,(1180,880,1935,1172)); note(d,(1925,895),'Слепая зона: пик 16:00 доходит последним',22,(207,34,46),anchor='ra')
zone(d,(5,58,850,130),(110,119,129),4); note(d,(20,94),'Левый верх – самая «дорогая» зона – пустует',22,(110,119,129),anchor='lm')
note(d,(20,18),'F-паттерн: 1→2 полный проход по KPI · 3 короче по таблице · 4 вниз по левому краю',22)
Image.alpha_composite(im,ov).convert('RGB').save('images/qlik_do_F.png')

# ---------- Taipy ----------
im=Image.open('images/raw/taipy_do.png').convert('RGBA'); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
arrow(d,(50,122),(600,122))                                   # 1: заголовок
arrow(d,(50,262),(1000,262)); arrow(d,(1080,262),(1960,262))  # 2→3: проход по KPI, якорь 408
arrow(d,(50,398),(1000,398))                                  # 4: шапка и первая строка таблицы (короче)
arrow(d,(24,430),(24,900))                                    # 5: вниз по левому краю
badge(d,(50,122),1,r=26); badge(d,(50,262),2,r=26); badge(d,(1040,262),3,r=26); badge(d,(50,398),4,r=26); badge(d,(24,925),5,r=24)
zone(d,(1320,935,1975,1310)); note(d,(1965,965),'Слепая зона: пик 16:00',22,(207,34,46),anchor='ra')
note(d,(1470,52),'фильтр – справа вверху ✓',20,(110,119,129),anchor='rm')
note(d,(660,122),'F-паттерн: 1 заголовок · 2→3 KPI · 4 таблица · 5 вниз по левому краю',22,anchor='lm')
Image.alpha_composite(im,ov).convert('RGB').save('images/taipy_do_F.png')

# ---------- Qlik ПОСЛЕ ----------
im=Image.open('images/raw/qlik_posle.png').convert('RGBA'); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
arrow(d,(525,172),(1955,172))                      # 1→2: старт на статусе, проход по KPI
arrow(d,(48,372),(1120,372))                       # 3: шапка таблицы (короче)
arrow(d,(14,410),(14,990))                         # 4: вниз по левому краю
badge(d,(230,172),1); badge(d,(1650,172),2); badge(d,(48,372),3,r=26); badge(d,(14,1010),4,r=24)
note(d,(405,240),'старт: статус – крупнее всех и единственный цвет',20,anchor='ma')
note(d,(1795,205),'пик 16:00 вынесен в KPI ✓',20,anchor='ma')
note(d,(1290,226),'фильтр справа вверху ✓',20,(110,119,129),anchor='rm')
zone(d,(1300,1020,1980,1535),(110,119,129),4)
note(d,(1965,1112),'правый низ – только детали для разбора',20,(110,119,129),anchor='ra')
note(d,(610,1060),'F-паттерн: 1→2 полный проход по KPI · 3 короче по таблице · 4 вниз по левому краю',22,anchor='lm')
Image.alpha_composite(im,ov).convert('RGB').save('images/qlik_posle_F.png')

# ---------- Taipy ПОСЛЕ ----------
im=Image.open('images/raw/taipy_posle.png').convert('RGBA'); ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
arrow(d,(300,118),(1650,118))
arrow(d,(50,288),(1000,288))
arrow(d,(24,320),(24,860))
badge(d,(262,118),1); badge(d,(1560,118),2); badge(d,(50,288),3,r=26); badge(d,(24,885),4,r=24)
note(d,(330,172),'старт: статус – крупнее всех и единственный цвет',20,anchor='la')
note(d,(1645,11),'пик 16:00 вынесен в KPI ✓',15,anchor='ra')
note(d,(1655,168),'фильтр справа вверху ✓',20,(110,119,129),anchor='rm')
zone(d,(1310,915,1990,1300),(110,119,129),4)
note(d,(1975,995),'правый низ – только детали',20,(110,119,129),anchor='ra')
note(d,(560,945),'F-паттерн: 1→2 полный проход по KPI · 3 таблица · 4 вниз по левому краю',22,anchor='lm')
Image.alpha_composite(im,ov).convert('RGB').save('images/taipy_posle_F.png')
