// Desktop workbook authoring; source records remain in step1/backbones.json.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile, FileBlob } from '@oai/artifact-tool';

const root = path.resolve(process.argv[2] || '.');
if (process.argv[3] === 'inference') {
  await exportInference(root);
  process.exit(0);
}
if (process.argv[3] === 'training-check') {
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'results.xlsx')));
  const sheet=wb.worksheets.getItem('Training');
  const original=sheet.getRange('H3').values;
  const base=sheet.getRange('H2').values[0][0];
  for(const input of [base+0.01,0,null]) {
    sheet.getRange('H3').values=[[input]];
    wb.recalculate();
    const actual=sheet.getRange('I3').values[0][0];
    if(input===null ? actual!=='' : Math.abs(actual-100*(input-base))>1e-10)
      throw new Error('Training delta formula failed blank/zero/positive-input check');
  }
  sheet.getRange('H3').values=original;
  wb.recalculate();
  console.log('Training formula verified for positive F1, zero F1 and blank F1; no file exported.');
  process.exit(0);
}
if (process.argv[3] === 'training') {
  await exportTraining(root);
  process.exit(0);
}
const rows = JSON.parse(await fs.readFile(path.join(root, 'step1/backbones.json'), 'utf8'));
const columns = ['exp_id','backbone','family','pretrained_tag','seed','epochs','batch_size',
  'params_m','gmacs','macro_f1_val','top1_val','best_epoch','train_seconds_epoch',
  'train_val_seconds_epoch','latency_ms_batch1_fp32','gpu','status','config_path','curve_path'];
const labels = ['exp_id','Backbone','Họ','Tag pretrained','Seed','Epoch','Batch',
  'Params (M)','GMAC','Macro-F1 val','Top-1 val','Best epoch','Train (s/epoch)',
  'Train + val (s/epoch)','Latency FP32 (ms)','GPU','Trạng thái','Cấu hình nguồn','Biểu đồ nguồn'];
const outputPath=path.join(root,'results.xlsx');
const exists=await fs.stat(outputPath).then(()=>true,()=>false);
const wb = exists ? await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath)) : Workbook.create();
let sheet;
try { sheet=wb.worksheets.getItem('Backbones'); }
catch { sheet=wb.worksheets.add('Backbones'); }
sheet.getUsedRange()?.clear({applyTo:'all'});
sheet.deleteAllDrawings();
sheet.showGridLines = false;
sheet.getRange('A1:S6').values = [labels, ...rows.map(r => columns.map(c => r[c] ?? null))];
sheet.getRange('A1:S11').format.font = {name:'Arial', size:10, color:'#243B53'};
sheet.getRange('A1:S6').format.verticalAlignment = 'center';
sheet.getRange('A1:S1').format = {fill:'#243B53', font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
  wrapText:true, horizontalAlignment:'center', verticalAlignment:'center', rowHeight:42};
sheet.getRange('A2:S6').format.rowHeight = 24;
sheet.getRange('A1:A6').format.columnWidthPx = 70;
sheet.getRange('B1:B6').format.columnWidthPx = 225;
sheet.getRange('C1:C6').format.columnWidthPx = 110;
sheet.getRange('D1:D6').format.columnWidthPx = 110;
sheet.getRange('E1:G6').format.columnWidthPx = 65;
sheet.getRange('H1:I6').format.columnWidthPx = 80;
sheet.getRange('J1:K6').format.columnWidthPx = 110;
sheet.getRange('L1:L6').format.columnWidthPx = 90;
sheet.getRange('M1:N6').format.columnWidthPx = 120;
sheet.getRange('O1:O6').format.columnWidthPx = 130;
sheet.getRange('P1:P6').format.columnWidthPx = 95;
sheet.getRange('Q1:Q6').format.columnWidthPx = 225;
sheet.getRange('R1:S6').format.columnWidthPx = 340;
sheet.getRange('E2:O6').format.horizontalAlignment = 'right';
sheet.getRange('H2:I6').setNumberFormat('0.000');
sheet.getRange('J2:K6').setNumberFormat('0.00%');
sheet.getRange('M2:N6').setNumberFormat('0.00');
sheet.getRange('O2:O6').setNumberFormat('0.000');
sheet.getRange('Q2:Q6').conditionalFormats.add('containsText', {
  text:'Chưa chạy', format:{fill:'#FFF3CD',font:{color:'#805B10'}}});
sheet.getRange('A9').values = [['T00: 10 epoch, batch 32, seed 0. Chỉ val; một seed/model.']];
const incomplete=rows.filter(r=>r.status!=='Đủ kết quả').length;
sheet.getRange('A10').values = [[incomplete ? `${incomplete} backbone còn thiếu kết quả; ô chưa đo được để trống.` :
  'Đủ năm backbone; latency sơ bộ FP32, batch 1, 10 warmup + một forward.']];
sheet.getRange('A11').values = [['Nguồn số liệu: runs/B0x/seed0/config.json, history.csv, summary.json, latency_preliminary.json.']];
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
wb.recalculate();
const check = await wb.inspect({kind:'table', range:'Backbones!A1:S6',include:'values,formulas',
  tableMaxRows:6,tableMaxCols:19,maxChars:1800});
console.log(check.ndjson);
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',
  options:{useRegex:true,maxResults:20},maxChars:400});
console.log(errors.ndjson);
const previewDir=path.join(root,'runs/step1_validation');
await fs.mkdir(previewDir,{recursive:true});
for (const [label,range] of [['metrics','A1:N6'],['provenance','O1:S6']]) {
  const png=await wb.render({sheetName:'Backbones',range,scale:1,format:'png'});
  await fs.writeFile(path.join(previewDir,`backbones_${label}.png`),new Uint8Array(await png.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log('Saved results.xlsx; missing experiments remain blank.');

async function exportTraining(root) {
  const rows = JSON.parse(await fs.readFile(path.join(root,'step2/ablations.json'),'utf8'));
  const columns = JSON.parse(await fs.readFile(path.join(root,'step2/workbook_schema.json'),'utf8'));
  const outputPath = path.join(root,'results.xlsx');
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const before = JSON.stringify(wb.worksheets.getItem('Backbones').getRange('A1:S11').values);
  const previewDir = path.join(root,'runs/step2_validation');
  await fs.mkdir(previewDir,{recursive:true});
  const old = await wb.render({sheetName:'Backbones',range:'A1:N6',scale:1,format:'png'});
  await fs.writeFile(path.join(previewDir,'backbones_before.png'),new Uint8Array(await old.arrayBuffer()));
  let sheet;
  try { sheet=wb.worksheets.getItem('Training'); }
  catch { sheet=wb.worksheets.add('Training'); }
  sheet.getUsedRange()?.clear({applyTo:'all'});
  sheet.deleteAllDrawings();
  sheet.showGridLines=false;
  const labels=columns.map(c => ({macro_f1_val:'Macro-F1 val',delta_f1_pp:'Δ F1 (pp)',
    top1_val:'Top-1 val',ece_val:'ECE val',nll_val:'NLL val',balanced_acc_val:'Balanced acc val',
    f1_chinee_apple:'F1 Chinee Apple',f1_snake_weed:'F1 Snake Weed',
    train_seconds_epoch:'Train (s/epoch)',changed_fields:'Khác T00',axis:'Trục',variant:'Biến thể',
    status:'Trạng thái',notes:'Ghi chú',config_path:'Cấu hình nguồn',curve_path:'Biểu đồ nguồn'}[c]||c));
  sheet.getRange('A1:AB9').values=[labels,...rows.map(r=>columns.map(c=>r[c]??null))];
  sheet.getRange('A1:AB9').format={font:{name:'Arial',size:10,color:'#243B53'},verticalAlignment:'center',columnWidthPx:120};
  sheet.getRange('A1:AB1').format={fill:'#243B53',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    wrapText:true,horizontalAlignment:'center',rowHeight:44};
  sheet.getRange('A2:AB9').format.rowHeight=40;
  sheet.getRange('A1:A9').format.columnWidthPx=65;
  sheet.getRange('B1:B9').format.columnWidthPx=150;
  sheet.getRange('C1:C9').format.columnWidthPx=55;
  sheet.getRange('D1:D9').format.columnWidthPx=145;
  sheet.getRange('E1:E9').format.columnWidthPx=245;
  sheet.getRange('E2:G9').format.wrapText=true;
  sheet.getRange('F1:G9').format.columnWidthPx=150;
  sheet.getRange('H2:Q9').format.horizontalAlignment='right';
  sheet.getRange('H2:H9').setNumberFormat('0.00%');
  sheet.getRange('J2:J9').setNumberFormat('0.00%');
  sheet.getRange('M2:O9').setNumberFormat('0.00%');
  sheet.getRange('I2:I9').setNumberFormat('+0.000;-0.000;0.000');
  sheet.getRange('K2:L9').setNumberFormat('0.0000');
  sheet.getRange('Q2:Q9').setNumberFormat('0.00');
  sheet.getRange('R1:R9').format.columnWidthPx=145;
  sheet.getRange('S1:X9').format.columnWidthPx=120;
  sheet.getRange('Y1:Y9').format.columnWidthPx=145;
  sheet.getRange('Z1:Z9').format.columnWidthPx=175;
  sheet.getRange('AA1:AB9').format.columnWidthPx=360;
  for(let r=2;r<=9;r++) sheet.getRange(`I${r}`).formulas=[[`=IF(ISNUMBER(H${r}),100*(H${r}-$H$2),"")`]];
  sheet.getRange('R2:R9').conditionalFormats.add('containsText',{
    text:'Chưa chạy',format:{fill:'#FFF3CD',font:{color:'#805B10'}}});
  const complete=rows.every(r=>r.status==='Đủ kết quả');
  if(complete) {
    const winner=rows.reduce((best,row)=>row.macro_f1_val>best.macro_f1_val?row:best,rows[0]);
    const selectedRow=rows.indexOf(winner)+2;
    sheet.getRange(`A${selectedRow}:AB${selectedRow}`).format.fill='#E4F0E9';
    sheet.getRange(`H${selectedRow}:I${selectedRow}`).format.font={name:'Arial',size:10,bold:true,color:'#243B53'};
    sheet.getRange(`Z${selectedRow}`).values=[['Chọn theo F1 val; một seed']];
  }
  sheet.getRange('A12:G12').merge();
  sheet.getRange('A12').values=[['T00 dùng lại B03; seed 0, 10 epoch, batch 32. Chỉ val.']];
  sheet.getRange('A13:G13').merge();
  sheet.getRange('A13').values=[[complete?
    'Đủ T00–T07. Sáu ablation và một kết hợp; dòng tô màu là công thức chọn bằng macro-F1 val.':
    'T01–T06: một yếu tố/run; T07: kết hợp chọn bằng val. Các ô chưa chạy để trống.']];
  sheet.getRange('A14:G14').merge();
  sheet.getRange('A14').values=[['Δ tính theo điểm phần trăm. Chưa có std giữa seed; không suy ra ưu thế từ chênh lệch nhỏ.']];
  sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(3);
  wb.recalculate();
  const values=sheet.getRange('H2:O9').values;
  if(Math.abs(values[0][0]-rows[0].macro_f1_val)>1e-12 || values[0][1]!==0)
    throw new Error('T00 metric/formula mismatch');
  rows.forEach((row,index)=>{
    if(row.macro_f1_val==null) {
      if(values[index][0]!=null || !['',null,undefined].includes(values[index][1]))
        throw new Error('Unrun recipe must have blank F1 and delta');
    } else if(Math.abs(values[index][0]-row.macro_f1_val)>1e-12 ||
              Math.abs(values[index][1]-row.delta_f1_pp)>1e-10) {
      throw new Error('Completed recipe metric/formula mismatch');
    }
  });
  if(before!==JSON.stringify(wb.worksheets.getItem('Backbones').getRange('A1:S11').values))
    throw new Error('Unrelated Backbones values changed');
  console.log((await wb.inspect({kind:'table',range:'Training!H1:R9',include:'values,formulas',
    tableMaxRows:9,tableMaxCols:11,maxChars:1500})).ndjson);
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',
    options:{useRegex:true,maxResults:20},maxChars:400})).ndjson);
  for(const [label,range] of [['plan','A1:G14'],['metrics','H1:R9'],['recipe','S1:AB9']]) {
    const png=await wb.render({sheetName:'Training',range,scale:1,format:'png'});
    await fs.writeFile(path.join(previewDir,`training_${label}.png`),new Uint8Array(await png.arrayBuffer()));
  }
  const backupPath=path.join(previewDir,'results_before_step2.xlsx');
  if(!await fs.stat(backupPath).then(()=>true,()=>false)) await fs.copyFile(outputPath,backupPath);
  const output=await SpreadsheetFile.exportXlsx(wb);
  await output.save(outputPath);
  console.log('Saved Training; T00 verified, unrun metrics blank, Backbones preserved.');
}

async function exportInference(root) {
  const outputPath=path.join(root,'results.xlsx');
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const preserved=['Backbones','Training'].map(name=>[name,JSON.stringify(wb.worksheets.getItem(name).getUsedRange().values)]);
  const directory=path.join(root,'runs/step3_validation');
  await fs.mkdir(directory,{recursive:true});
  for(const [name,range] of [['Backbones','A1:N6'],['Training','H1:R9']]) {
    const png=await wb.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(directory,`${name.toLowerCase()}_before.png`),new Uint8Array(await png.arrayBuffer()));
  }
  const labels={method:'Phương pháp',checkpoint:'Checkpoint',k:'K',img_size:'Crop size',input_size:'Input size',
    temperature:'Temperature T',aggregation:'Gộp view',macro_f1_val:'Macro-F1 val',top1_val:'Top-1 val',
    ece_val:'ECE val',nll_val:'NLL val',p50_ms:'p50 (ms)',p95_ms:'p95 (ms)',p99_ms:'p99 (ms)',mean_ms:'Mean (ms)',
    relative_cost:'Chi phí × I00',images_per_s_batch32:'Ảnh/s batch 32',images_per_s:'Ảnh/s',p95_le_100ms:'p95 ≤ 100 ms',
    status:'Trạng thái',prediction_path:'Dự đoán val',iterations:'Lần đo',preprocessing_included:'Gồm tiền xử lý',
    transfer_included:'Gồm transfer',timing_scope:'Phạm vi đo',source_path:'Nguồn số đo'};
  const letter=i=>String.fromCharCode(65+i); // These two schemas have 21 columns.
  for(const [name,records,schema] of [['Inference','inference.json','inference_schema.json'],['Latency','latency.json','latency_schema.json']]) {
    const rows=JSON.parse(await fs.readFile(path.join(root,'step3',records),'utf8'));
    const columns=JSON.parse(await fs.readFile(path.join(root,'step3',schema),'utf8'));
    if(columns.length>26) throw new Error('Update column-letter helper for larger schema');
    let sheet;
    try {sheet=wb.worksheets.getItem(name);} catch {sheet=wb.worksheets.add(name);}
    sheet.getUsedRange()?.clear({applyTo:'all'});sheet.deleteAllDrawings();sheet.showGridLines=false;
    const last=Math.max(2,rows.length+1),end=letter(columns.length-1);
    const data=[columns.map(c=>labels[c]||c),...rows.map(r=>columns.map(c=>r[c]??null))];
    if(!rows.length) data.push(columns.map(()=>null));
    sheet.getRange(`A1:${end}${last}`).values=data;
    sheet.getRange(`A1:${end}${last}`).format={font:{name:'Arial',size:10,color:'#243B53'},columnWidthPx:115,verticalAlignment:'center'};
    sheet.getRange(`A1:${end}1`).format={fill:'#243B53',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
      wrapText:true,horizontalAlignment:'center',rowHeight:45};
    sheet.getRange(`A2:${end}${last}`).format.rowHeight=36;
    columns.forEach((column,index)=>{
      const c=letter(index),body=sheet.getRange(`${c}2:${c}${last}`);
      if(column==='exp_id')sheet.getRange(`${c}1:${c}${last}`).format.columnWidthPx=65;
      if(['method','status','timing_scope'].includes(column)) {
        sheet.getRange(`${c}1:${c}${last}`).format.columnWidthPx=column==='timing_scope'?450:225;body.format.wrapText=true;
      }
      if(column==='checkpoint'||column.endsWith('_path'))sheet.getRange(`${c}1:${c}${last}`).format.columnWidthPx=335;
      if(['macro_f1_val','top1_val'].includes(column))body.setNumberFormat('0.00%');
      else if(['ece_val','nll_val'].includes(column))body.setNumberFormat('0.00000');
      else if(column==='temperature')body.setNumberFormat('0.000000');
      else if(column.endsWith('_ms'))body.setNumberFormat('0.000');
      else if(column.startsWith('images_per_s'))body.setNumberFormat('0.0');
      if(column==='relative_cost') {
        body.setNumberFormat('0.00"×"');
        for(let r=2;r<=last;r++)sheet.getRange(`${c}${r}`).formulas=[[`=IF(AND(ISNUMBER(N${r}),ISNUMBER($N$2),$N$2>0),N${r}/$N$2,"")`]];
      }
      if(column==='p95_le_100ms')for(let r=2;r<=last;r++)sheet.getRange(`${c}${r}`).formulas=[[`=IF(ISNUMBER(O${r}),IF(O${r}<=100,"Đạt","Không đạt"),"Chờ đo")`]];
    });
    sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(2);
    const footer=last+3;
    for(let r=footer;r<footer+3;r++)sheet.getRange(`A${r}:G${r}`).merge();
    sheet.getRange(`A${footer}`).values=[[name==='Inference'?
      'T05, seed 0; chỉ val. Chọn sau khi đủ dự đoán và latency thật trên T4.':
      rows.length?'Raw samples: step3/Ixx/latency_batch1/32.json; batch 1 và 32.':'Chưa có số đo T4. Không thay bằng latency sơ bộ Bước 1 hoặc GPU máy cá nhân.']];
    sheet.getRange(`A${footer+1}`).values=[['10 warmup + 100 lần đo; synchronize trước/sau. Input sẵn trên GPU; gồm TTA/model/softmax/T.']];
    sheet.getRange(`A${footer+2}`).values=[['Không gồm decode/PIL/H2D/D2H. p95≤100 ms là ngưỡng pipeline GPU; chưa phải toàn bộ camera/robot.']];
  }
  wb.recalculate();
  const inf=wb.worksheets.getItem('Inference');
  // Boundary checks in memory; restore every temporary timing before export.
  const saved=[inf.getRange('N2').values,inf.getRange('N3').values,inf.getRange('O3').values];
  inf.getRange('N2').values=[[5]];inf.getRange('N3').values=[[15]];inf.getRange('O3').values=[[101]];
  wb.recalculate();
  if(inf.getRange('Q3').values[0][0]!==3||inf.getRange('S3').values[0][0]!=='Không đạt')throw new Error('Cost/threshold formula failed');
  inf.getRange('O3').values=[[100]];wb.recalculate();
  if(inf.getRange('S3').values[0][0]!=='Đạt')throw new Error('100 ms boundary failed');
  inf.getRange('N2').values=saved[0];inf.getRange('N3').values=saved[1];inf.getRange('O3').values=saved[2];
  wb.recalculate();
  const rows=JSON.parse(await fs.readFile(path.join(root,'step3/inference.json'),'utf8'));
  rows.forEach((r,index)=>{
    const actual=inf.getRange(`J${index+2}:R${index+2}`).values[0];
    if(r.macro_f1_val!=null&&Math.abs(actual[0]-r.macro_f1_val)>1e-12)throw new Error('Inference F1 mismatch');
    if(r.p95_ms==null&&actual[5]!=null)throw new Error('Unmeasured latency must remain blank');
  });
  for(const [name,before]of preserved)if(JSON.stringify(wb.worksheets.getItem(name).getUsedRange().values)!==before)throw new Error(`Changed ${name}`);
  console.log((await wb.inspect({kind:'table',range:'Inference!I1:T10',include:'values,formulas',tableMaxRows:10,tableMaxCols:12,maxChars:1200})).ndjson);
  console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:400})).ndjson);
  for(const[name,label,range]of [['Inference','methods','A1:I10'],['Inference','metrics','J1:T10'],['Latency','pending','A1:N5']]) {
    const png=await wb.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(directory,`${name.toLowerCase()}_${label}.png`),new Uint8Array(await png.arrayBuffer()));
  }
  const backup=path.join(directory,'results_before_step3.xlsx');
  if(!await fs.stat(backup).then(()=>true,()=>false))await fs.copyFile(outputPath,backup);
  const output=await SpreadsheetFile.exportXlsx(wb);await output.save(outputPath);
  console.log('Saved Inference/Latency. B/T sheets preserved; no latency invented.');
}
