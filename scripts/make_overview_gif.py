"""Render an 18-second paper overview with Pillow; no results or weights needed.

Concept: one reused denoiser, stage-aware compression, three separate measurements.
Token text is illustrative. Diagram widths do not represent measured quantities.
"""
from pathlib import Path
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS, SECONDS = 1100, 620, 10, 18
BG, PANEL, WHITE = '#0a1222', '#121f34', '#eef4ff'
MUTED, CYAN, GOLD, PURPLE = '#b0bed6', '#56ddce', '#ffd17c', '#baa5ff'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--font', type=Path, default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    args = ap.parse_args()
    if not args.font.exists():
        raise SystemExit('DejaVu Sans not found; pass --font /path/to/font.ttf')
    fonts = {size: ImageFont.truetype(str(args.font), size) for size in (13, 15, 16, 18, 20, 22, 24, 28, 34, 38)}
    base = Image.new('RGB', (W, H), BG)
    b = ImageDraw.Draw(base)
    for y in range(H):
        v = int(9 * (1 - y / H))
        b.line((0,y,W,y), fill=(10+v,18+v,34+v))
    for x in range(30,W,40):
        for y in range(135,560,40):
            b.ellipse((x,y,x+1,y+1), fill='#25324a')

    def text(d, xy, value, size=20, color=WHITE):
        d.text(xy, value, font=fonts[size], fill=color)

    def center(d, box, value, size=20, color=WHITE):
        x0,y0,x1,y1=box
        bounds=d.textbbox((0,0),value,font=fonts[size])
        text(d, ((x0+x1-(bounds[2]-bounds[0]))/2,(y0+y1-(bounds[3]-bounds[1]))/2-bounds[1]),value,size,color)

    def panel(d, box, outline='#2b3c58', fill=PANEL):
        d.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)

    def arrow(d, start, end, color=CYAN):
        x,y=start; a,z=end
        d.line((x,y,a,z),fill=color,width=3)
        d.polygon(((a,z),(a-11,z-6),(a-11,z+6)),fill=color)

    def frame(t):
        im=base.copy();d=ImageDraw.Draw(im)
        scene=min(2,int(t//6)); u=t-scene*6
        text(d,(42,24),'EMNLP 2026  /  INDUSTRY TRACK',15,CYAN)
        text(d,(42,55),'Quantizing Masked Diffusion Language Models',34)
        text(d,(42,104),'Storage–quality trade-offs and runtime diagnostics',20,MUTED)
        d.line((42,145,1058,145),fill='#2b3c58',width=1)
        if scene==0:
            text(d,(42,172),'01   The same denoiser. Many chances to refine.',24)
            text(d,(42,210),'Weight errors are reused across the entire decoding trajectory.',18,MUTED)
            tokens=['A','compact','model','can','refine','text']
            for row in range(3):
                y=267+row*68
                text(d,(48,y+16),['MASKED','REFINING','FILLED'][row],13,MUTED)
                visible=[0,3,6][row]
                if row==1:visible=min(5,1+int(u/1.2))
                for j,v in enumerate(tokens):
                    x=175+j*128
                    show=j<visible
                    panel(d,(x,y,x+116,y+46),CYAN if show else '#344766')
                    center(d,(x,y,x+116,y+46),v if show else '[MASK]',16,WHITE if show else MUTED)
            panel(d,(42,495,1058,550))
            center(d,(42,495,1058,550),'Question: can lower-bit weights preserve quality across repeated denoising?',18,GOLD)
        elif scene==1:
            text(d,(42,172),'02   Calibrate stages. Encode a base + residual.',24)
            text(d,(42,210),'TurboLDM reference: 24 calibration texts, separate from main task evaluation.',18,MUTED)
            boxes=[(42,270,342,423),(400,270,700,423),(758,270,1058,423)]
            titles=['ROTATE + QUANTIZE','RANK SENSITIVITY','ALLOCATE RESIDUAL']
            subs=['Hadamard blocks + codebook','Layer × masking-stage pairs','3-bit base + 1-bit correction']
            for k,box in enumerate(boxes):
                panel(d,box,[CYAN,PURPLE,GOLD][k]);center(d,(box[0],280,box[2],317),titles[k],16,[CYAN,PURPLE,GOLD][k])
                center(d,(box[0],382,box[2],414),subs[k],15,MUTED)
            for j in range(8):
                x=64+j*32
                height=18+int(22*(1+math.sin(j*1.8+u*1.3))/2)
                d.rounded_rectangle((x,371-height,x+19,371),radius=4,fill=CYAN)
            for r in range(3):
                for c in range(6):
                    x,y=430+c*41,329+r*14
                    d.rounded_rectangle((x,y,x+30,y+8),radius=2,fill=PURPLE if (r*5+c*3)%7<2 else '#34415b')
            for j in range(7):
                x=783+j*36
                d.rounded_rectangle((x,332,x+27,371),radius=4,fill=CYAN)
                if j in (1,4):d.rounded_rectangle((x,321,x+27,329),radius=3,fill=GOLD)
            arrow(d,(352,347),(390,347));arrow(d,(710,347),(748,347))
            panel(d,(42,459,1058,550))
            center(d,(42,464,1058,504),'Compression is a trade-off, not a guarantee of equal quality.',22)
            center(d,(42,503,1058,542),'Extra denoising partly helps GSM8K; multiple-choice accuracy stays largely flat.',16,MUTED)
        else:
            text(d,(42,172),'03   Measure three different deployment questions.',24)
            labels=[('STORAGE','How small is the','representation?','TurboLDM-(3+1)','Compact; quality loss',CYAN),
                    ('QUALITY','How well does the','model answer?','AWQ-4','Closest tested 4-bit point',PURPLE),
                    ('RUNTIME','How much live GPU','memory is used?','NF4','Lower VRAM in tested runtime',GOLD)]
            for k,(label,a,b,method,sub,color) in enumerate(labels):
                x=42+k*344
                panel(d,(x,234,x+328,467),color)
                text(d,(x+21,253),label,15,color)
                text(d,(x+21,294),a,22);text(d,(x+21,325),b,22)
                text(d,(x+21,383),method,28,color)
                text(d,(x+21,430),sub,15,MUTED)
            panel(d,(42,490,1058,550))
            center(d,(42,490,1058,519),'Smaller serialized storage does not imply lower live VRAM.',20)
            center(d,(42,519,1058,548),'NF4 memory gains depend on this runtime; other optimized kernels exist.',15,MUTED)
        text(d,(42,578),'Pawan Kumar · Neel Mishra',15,MUTED)
        text(d,(635,578),'github.com/misterpawan/turbo-quant-paper',15,MUTED)
        for k in range(3):
            x=42+k*344
            d.rounded_rectangle((x,558,x+328,561),radius=1,fill='#2b3c58')
            if k<scene:d.rounded_rectangle((x,558,x+328,561),radius=1,fill=CYAN)
            if k==scene:d.rounded_rectangle((x,558,x+max(1,int(328*u/6)),561),radius=1,fill=CYAN)
        return im

    out=ROOT/'assets';out.mkdir(exist_ok=True)
    frames=[frame(i/FPS) for i in range(FPS*SECONDS)]
    # A shared palette prevents palette flicker in otherwise static cards.
    palette=frames[145].quantize(colors=128)
    frames=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    frames[0].save(out/'paper-overview.gif',save_all=True,append_images=frames[1:],
                   duration=1000//FPS,loop=0,optimize=True,disposal=2)
    frame(15).save(out/'paper-overview.png')
    sheet=Image.new('RGB',(W,3*H),BG)
    for i,t in enumerate((3,9,15)):sheet.paste(frame(t),(0,i*H))
    sheet.save(out/'overview-contact-sheet.png')
    print(f'Wrote {SECONDS}s, {FPS}fps, {W}×{H} overview: {out / "paper-overview.gif"}')


if __name__=='__main__':main()
