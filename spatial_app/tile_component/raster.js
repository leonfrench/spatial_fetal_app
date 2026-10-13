/* Pure numeric raster operations, also exercised by the Node tests. */
(function(root) {
function aggregate(g, scores, factor, mode, aucs) {
 const w=Math.ceil(g.width/factor), h=Math.ceil(g.height/factor), n=w*h;
 const sums=new Float64Array(n), measured=new Uint32Array(n), occupied=new Uint32Array(n);
 const representatives=new Int32Array(n);representatives.fill(-1);
 const distances=new Float64Array(n);distances.fill(Infinity);
 const categories=new Uint16Array(n), votes=mode==='Cell type'?new Uint32Array(n*g.labels.length):null;
 for(let i=0;i<g.positions.length;i++) {
  const p=g.positions[i], b=Math.floor(Math.floor(p/g.width)/factor)*w+Math.floor((p%g.width)/factor);
  occupied[b]++;
  // Choose the occupied bin nearest this block's center, independent of scores.
  // Stable input order breaks distance ties, including at partial edge blocks.
  const bx=b%w,by=Math.floor(b/w);
  const cx=(bx*factor+Math.min((bx+1)*factor,g.width)-1)/2;
  const cy=(by*factor+Math.min((by+1)*factor,g.height)-1)/2;
  const distance=(p%g.width-cx)**2+(Math.floor(p/g.width)-cy)**2;
  if(distance<distances[b]){distances[b]=distance;representatives[b]=i;}
  const c=g.categories[i];
  if(votes) votes[b*g.labels.length+c]++;
  const value=mode==='Tissue AUROC'?aucs[g.labels[c]]:scores[i];
  if(Number.isFinite(value)) {sums[b]+=value;measured[b]++;}
 }
 const values=new Float64Array(n);values.fill(NaN);
 for(let b=0;b<n;b++) {
  if(measured[b])values[b]=sums[b]/measured[b];
  if(votes && occupied[b]) {
   let best=0;
   for(let c=0;c<g.labels.length;c++)if(votes[b*g.labels.length+c]>best){best=votes[b*g.labels.length+c];categories[b]=c;}
  }
 }
 // Display-only neighboring preview. No values are written back to scores.
 const preview=new Uint8Array(n);
 if(mode==='Gene-set AUROC') {
  const queue=new Int32Array(n), visited=new Uint8Array(n);let head=0,tail=0;
  for(let b=0;b<n;b++)if(measured[b]){queue[tail++]=b;visited[b]=1;}
  function visit(from,to){if(!visited[to]){visited[to]=1;values[to]=values[from];queue[tail++]=to;}}
  while(head<tail){const b=queue[head++],x=b%w;if(x)visit(b,b-1);if(x+1<w)visit(b,b+1);if(b>=w)visit(b,b-w);if(b+w<n)visit(b,b+w);}
  for(let b=0;b<n;b++)if(occupied[b]&&!measured[b]&&Number.isFinite(values[b]))preview[b]=1;
 }
 return {w,h,factor,values,measured,occupied,categories,preview,representatives};
}
function hoverText(g,scores,raster,block,mode,aucs) {
 const id=raster.representatives[block];
 if(id<0)return '';
 const tissue=g.labels[g.categories[id]];
 const value=mode==='Tissue AUROC'?aucs[tissue]:scores[id];
 return tissue+'\nAUROC: '+(Number.isFinite(value)?value.toFixed(4):'Not scored');
}
function factorForScale(scale,dpr=1){let f=1;while(f*2*scale*dpr<=1)f*=2;return f;}
root.SpatialRaster={aggregate,factorForScale,hoverText};
if(typeof module!=='undefined')module.exports=root.SpatialRaster;
})(globalThis);
