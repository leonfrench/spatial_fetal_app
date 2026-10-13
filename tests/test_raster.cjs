const assert=require('node:assert/strict');
const {aggregate,factorForScale}=require('../spatial_app/tile_component/raster.js');
const g={width:4,height:2,positions:new Uint32Array([0,1,4,7]),categories:new Uint16Array([0,1,1,0]),labels:['A','B']};
const scores=new Float64Array([.2,.6,NaN,.8]);
let r=aggregate(g,scores,2,'Gene-set AUROC',{});
assert.equal(r.values[0],.4);assert.equal(r.measured[0],2);assert.equal(r.occupied[0],3);
assert.equal(r.values[1],.8);
r=aggregate(g,scores,1,'Gene-set AUROC',{});
assert.equal(r.values[0],.2);assert.equal(r.occupied[2],0);assert.equal(r.preview[4],1);assert(Number.isNaN(scores[2]));
r=aggregate(g,scores,2,'Cell type',{});assert.equal(r.categories[0],1);
r=aggregate(g,scores,1,'Tissue AUROC',{A:.45,B:.55});assert.equal(r.values[4],.55);
assert.equal(factorForScale(.2),4);assert.equal(factorForScale(2),1);assert.equal(factorForScale(.2,2),2);
console.log('Raster aggregation, gaps, preview, category and zoom tests passed');
const {hoverText}=require('../spatial_app/tile_component/raster.js');
r=aggregate(g,scores,2,'Gene-set AUROC',{});
assert.equal(r.representatives[0],0); // Stable tie break, actual .2 rather than mean .4.
assert.equal(hoverText(g,scores,r,0,'Gene-set AUROC',{}),'A\nAUROC: 0.2000');
assert.equal(hoverText(g,scores,r,0,'Cell type',{}),'A\nAUROC: 0.2000');
assert.equal(hoverText(g,scores,r,0,'Tissue AUROC',{A:.45}),'A\nAUROC: 0.4500');
r=aggregate(g,scores,1,'Gene-set AUROC',{});
assert.equal(hoverText(g,scores,r,4,'Gene-set AUROC',{}),'B\nAUROC: Not scored');
assert.equal(hoverText(g,scores,r,2,'Gene-set AUROC',{}),'');
const centered={width:3,height:3,positions:new Uint32Array([0,4,8]),categories:new Uint16Array([0,1,0]),labels:['A','B']};
r=aggregate(centered,new Float64Array([.1,.5,.9]),4,'Gene-set AUROC',{});
assert.equal(r.representatives[0],1); // Center of a clipped edge block.
console.log('Representative-bin hover tests passed');
