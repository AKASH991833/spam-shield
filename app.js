/* spam-shield — Naive Bayes spam classifier, fully in-browser */
(function(){
'use strict';
const STOP=new Set('a an the is are was were i you he she it we they me him her us them my your his their our this that these those and or but if then so of to in on at for with by from as be been have has had do does did will would can could should not no yes'.split(' '));
function tokenize(text,useBigrams){
  let t=text.toLowerCase();
  const urls=t.match(/https?:\/\/\S+|www\.\S+|[a-z0-9-]+\.(com|net|org|in|top|site|biz|info|shop|click|app|co|gg|mobi)\b/g)||[];
  t=t.replace(/https?:\/\/\S+|www\.\S+/g,' urltok ');
  t=t.replace(/[\u20B9$£€]\s?\d[\d,]*(\.\d+)?/g,' moneytok ');
  t=t.replace(/\b\d{6}\b/g,' otptok ');
  t=t.replace(/\+?\d[\d\s-]{7,}\d/g,' phonetok ');
  let words=t.replace(/[^a-z0-9\u0900-\u097F\s]/g,' ').split(/\s+/).filter(w=>w.length>1&&!STOP.has(w));
  if(urls.length)words.push('haslink');
  if(/[A-Z]{4,}/.test(text))words.push('manycaps');
  if((text.match(/!/g)||[]).length>=2)words.push('bangs');
  if(useBigrams){const bi=[];for(let i=0;i<words.length-1;i++)bi.push(words[i]+'_'+words[i+1]);words=words.concat(bi)}
  return words;
}
class NB{
  constructor(){this.reset()}
  reset(){this.classCounts=[0,0];this.tokCounts=[{},{}];this.classTotal=[0,0];this.vocab=new Set()}
  train(docs,labels){
    this.reset();
    docs.forEach((toks,i)=>{const c=labels[i];this.classCounts[c]++;
      const seen=new Set(toks); // Bernoulli-flavoured multinomial: count once per doc
      seen.forEach(w=>{this.tokCounts[c][w]=(this.tokCounts[c][w]||0)+1;this.classTotal[c]++;this.vocab.add(w)})});
  }
  logProb(toks,c){
    const V=this.vocab.size,Nc=this.classTotal[c],alpha=0.5;
    let lp=Math.log(this.classCounts[c]/(this.classCounts[0]+this.classCounts[1]));
    toks.forEach(w=>{lp+=Math.log(((this.tokCounts[c][w]||0)+alpha)/(Nc+alpha*V))});
    return lp;
  }
  predict(text,opt){
    const toks=tokenize(text,opt&&opt.bigrams);
    const l1=this.logProb(toks,1),l0=this.logProb(toks,0);
    const p=1/(1+Math.exp(l0-l1));
    return {prob:p,toks};
  }
  contributions(text,opt){
    const toks=[...new Set(tokenize(text,opt&&opt.bigrams))];
    const V=this.vocab.size,alpha=0.5;
    return toks.map(w=>{
      const s1=Math.log(((this.tokCounts[1][w]||0)+alpha)/(this.classTotal[1]+alpha*V));
      const s0=Math.log(((this.tokCounts[0][w]||0)+alpha)/(this.classTotal[0]+alpha*V));
      return {w,score:s1-s0};
    }).filter(x=>isFinite(x.score)).sort((a,b)=>b.score-a.score);
  }
}
function shuffleSeeded(arr,seed){let s=seed;const a=arr.slice();for(let i=a.length-1;i>0;i--){s=(s*9301+49297)%233280;const j=Math.floor(s/233280*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a}
let model=new NB(),useBigrams=true,threshold=0.5,splitInfo=null;
function buildModel(){
  const spam=shuffleSeeded(SPAM_DATA,42),ham=shuffleSeeded(HAM_DATA,7);
  const sCut=Math.floor(spam.length*0.8),hCut=Math.floor(ham.length*0.8);
  const trainD=[],trainL=[],testD=[],testL=[];
  spam.forEach((x,i)=>{(i<sCut?trainD:testD).push(tokenize(x[0],useBigrams));(i<sCut?trainL:testL).push(1)});
  ham.forEach((x,i)=>{(i<hCut?trainD:testD).push(tokenize(x[0],useBigrams));(i<hCut?trainL:testL).push(0)});
  model.train(trainD,trainL);
  let tp=0,tn=0,fp=0,fn=0;
  testD.forEach((toks,i)=>{
    const l1=model.logProb(toks,1),l0=model.logProb(toks,0);
    const p=1/(1+Math.exp(l0-l1));
    const pred=p>=threshold?1:0;
    if(pred===1&&testL[i]===1)tp++;else if(pred===0&&testL[i]===0)tn++;else if(pred===1)fp++;else fn++;
  });
  const prec=tp/(tp+fp||1),rec=tp/(tp+fn||1);
  splitInfo={trainSize:trainD.length,testSize:testD.length,tp,tn,fp,fn,
    acc:(tp+tn)/testD.length,prec,rec,f1:2*prec*rec/(prec+rec||1)};
  renderMetrics();
}
function fmt(x){return (x*100).toFixed(1)+'%'}
function renderMetrics(){
  const s=splitInfo;if(!s)return;
  document.getElementById('m-acc').textContent=fmt(s.acc);
  document.getElementById('m-prec').textContent=fmt(s.prec);
  document.getElementById('m-rec').textContent=fmt(s.rec);
  document.getElementById('m-f1').textContent=fmt(s.f1);
  document.getElementById('m-split').textContent=`${s.trainSize} train / ${s.testSize} test (stratified 80/20, seeded)`;
  document.getElementById('cm').innerHTML=
    `<tr><th></th><th>Predicted spam</th><th>Predicted safe</th></tr>
     <tr><th>Actually spam</th><td class="cm-ok">${s.tp}</td><td class="cm-bad">${s.fn}</td></tr>
     <tr><th>Actually safe</th><td class="cm-bad">${s.fp}</td><td class="cm-ok">${s.tn}</td></tr>`;
  const top=model.tokCounts[1];
  const rows=Object.keys(top).map(w=>({w,c1:top[w]||0,c0:model.tokCounts[0][w]||0}))
    .filter(x=>x.c1>=3).map(x=>({...x,odds:(x.c1+0.5)/(x.c0+0.5)})).sort((a,b)=>b.odds-a.odds).slice(0,15);
  document.getElementById('topwords').innerHTML=rows.map((r,i)=>
    `<div class="tw-row"><span class="tw-rank">${i+1}</span><span class="tw-word">${r.w.replace('_',' + ')}</span><div class="meter" style="flex:1"><i style="width:${Math.min(100,r.odds/ (rows[0].odds)*100)}%"></i></div><span class="mono">${r.c1} spam / ${r.c0} safe</span></div>`).join('');
}
function classify(){
  const text=document.getElementById('msg').value.trim();
  if(!text){pkToast('Paste or type a message first');return}
  const {prob}=model.predict(text,{bigrams:useBigrams});
  const isSpam=prob>=threshold;
  document.getElementById('result').classList.remove('hidden');
  const pct=(prob*100);
  document.getElementById('verdict').innerHTML=isSpam
    ?`<span class="pill bad">Likely spam</span>`:`<span class="pill ok">Looks safe</span>`;
  document.getElementById('prob-num').textContent=pct.toFixed(1)+'%';
  document.getElementById('prob-meter').style.width=pct+'%';
  document.getElementById('prob-meter').style.background=isSpam?'var(--bad)':'var(--ok)';
  // highlighted message
  const contrib=model.contributions(text,{bigrams:useBigrams});
  const spammy=new Set(contrib.filter(c=>c.score>0.6).map(c=>c.w));
  const safe=new Set(contrib.filter(c=>c.score<-1.2).map(c=>c.w));
  const esc=text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  const hl=esc.split(/\b/).map(part=>{
    const lw=part.toLowerCase();
    if(spammy.has(lw))return `<mark class="hl-bad">${part}</mark>`;
    if(safe.has(lw))return `<mark class="hl-ok">${part}</mark>`;
    return part;
  }).join('');
  document.getElementById('highlighted').innerHTML=hl;
  const reasons=contrib.filter(c=>c.score>0.6&&!c.w.includes('_')).slice(0,8);
  document.getElementById('reasons').innerHTML=reasons.length
    ?reasons.map(r=>`<span class="pill bad">${r.w}</span>`).join(' ')
    :'<span class="pill ok">No strong spam signals found</span>';
  const safeTop=contrib.filter(c=>c.score<-1.2&&!c.w.includes('_')).slice(-6).map(r=>r.w);
  document.getElementById('safe-words').innerHTML=safeTop.length?safeTop.map(w=>`<span class="pill ok">${w}</span>`).join(' '):'';
}
function runBatch(){
  const raw=document.getElementById('batch-in').value.trim();
  if(!raw){pkToast('Paste messages, one per line');return}
  const lines=raw.split('\n').map(l=>l.trim()).filter(Boolean);
  const rows=lines.map(l=>{const {prob}=model.predict(l,{bigrams:useBigrams});return {l,prob,spam:prob>=threshold}});
  document.getElementById('batch-out').classList.remove('hidden');
  document.getElementById('batch-tbl').innerHTML=rows.map((r,i)=>
    `<tr><td>${i+1}</td><td>${r.l.length>90?r.l.slice(0,90)+'…':r.l}</td><td>${r.spam?'<span class="pill bad">Spam</span>':'<span class="pill ok">Safe</span>'}</td><td class="mono">${(r.prob*100).toFixed(0)}%</td></tr>`).join('');
  const n=rows.filter(r=>r.spam).length;
  document.getElementById('batch-sum').innerHTML=`<span class="pill bad">${n} spam</span> <span class="pill ok">${rows.length-n} safe</span> <span class="pill">${rows.length} total</span>`;
  document.getElementById('batch-dl').onclick=()=>pkDownload('spam-results.csv',pkCsv([['message','verdict','spam_probability'],...rows.map(r=>[r.l,r.spam?'spam':'safe',r.prob.toFixed(3)])]),'text/csv');
}
function loadSample(){
  const samples=[
    "Congratulations! Your mobile number has won ₹25 lakh in the KBC lottery. To claim, share your Aadhaar and pay ₹2,500 processing fee immediately.",
    "Hey, are we meeting at the library at 4 for the group project? Priya is bringing the printouts.",
    "URGENT: Your SBI account will be blocked today. Update KYC now at sbi-kyc-verify.info or lose access.",
    "Your cab is arriving in 3 minutes. Driver: Ramesh, white Swift UP32 AB 1234."
  ];
  document.getElementById('msg').value=samples[Math.floor(Math.random()*samples.length)];
  classify();
}
document.addEventListener('DOMContentLoaded',()=>{
  buildModel();
  document.getElementById('check').onclick=classify;
  document.getElementById('sample').onclick=loadSample;
  document.getElementById('batch-btn').onclick=runBatch;
  const th=document.getElementById('threshold');
  th.oninput=()=>{threshold=th.value/100;document.getElementById('th-val').textContent=th.value+'%';buildModel();if(document.getElementById('msg').value.trim())classify()};
  const bg=document.getElementById('bigrams');
  bg.onchange=()=>{useBigrams=bg.checked;buildModel();if(document.getElementById('msg').value.trim())classify()};
  pkTabs();
});
})();
