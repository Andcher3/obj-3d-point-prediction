from pathlib import Path
import cv2, numpy as np, json
videos=['10790317943435558913','gear_v2_9-wrist_front_rgb_img-0']; seeds=[42,43,44]
root=Path('outputs/matrix'); invid=Path('inputs/videos'); panels=[]
colors=[(0,255,255),(255,0,255),(0,200,0),(255,128,0),(0,0,255)]
for stem in videos:
 cap=cv2.VideoCapture(str(invid/f'{stem}.mp4')); cap.set(cv2.CAP_PROP_POS_FRAMES,161); ok,bg=cap.read(); cap.release();
 if not ok: raise RuntimeError(stem)
 row=[]
 d=root/stem/'seed_42'/stem; gt=np.load(d/'gt_tracks.npy'); vis=np.load(d/'gt_visibility.npy').astype(bool)
 series=[('GT',gt)]+[(f'seed {s}',np.load(root/stem/f'seed_{s}'/stem/'pred_tracks.npy')) for s in seeds]
 idx=np.linspace(0,gt.shape[1]-1,40,dtype=int)
 for title,tr in series:
  im=bg.copy()
  for qi in idx:
   pts=tr[:,qi].round().astype(int)
   okmask=vis[:,qi]
   pts=pts[okmask]
   if len(pts)<2: continue
   for a,b in zip(pts[:-1],pts[1:]): cv2.line(im,tuple(a),tuple(b),(0,255,255),1,cv2.LINE_AA)
   cv2.circle(im,tuple(pts[-1]),2,(0,0,255),-1,cv2.LINE_AA)
  cv2.rectangle(im,(0,0),(240,34),(255,255,255),-1); cv2.putText(im,f'{stem[:18]} | {title}',(8,23),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
  row.append(im)
 panels.append(np.hstack(row))
out=np.vstack(panels); cv2.imwrite('analysis/seed_comparison.png',out)
print('WROTE analysis/seed_comparison.png',out.shape)
