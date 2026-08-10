#!/usr/bin/env python3
"""Mechanical consistency guard for the audited PaperD evidence bundle."""
import csv, json, math, re, statistics as st, sys
from pathlib import Path

fail = []
def close(label, got, want, tol=5e-7):
    if abs(got-want)>tol: fail.append(f"{label}: {got} != {want}")

fac=list(csv.DictReader(open("audited_factorial_results.csv",encoding="utf-8")))
if len(fac)!=16: fail.append(f"factorial cells: {len(fac)} != 16")
keys={(r['mask'],r['lambda'],int(r['lr_multiplier'])) for r in fac}
expected={(m,l,x) for m in ('Dense','Top-0.4') for l in ('1.0','0.1') for x in (1,5,10,20)}
if keys!=expected: fail.append("factorial design is incomplete or duplicated")
for r in fac:
    vals=[float(r[f"seed{s}"]) for s in (42,43,44)]
    close(f"{r['mask']}/{r['lambda']}/{r['lr_multiplier']} mean",float(r['mean']),st.mean(vals))
    close(f"{r['mask']}/{r['lambda']}/{r['lr_multiplier']} sd",float(r['sample_std']),st.stdev(vals))

def cell(mask,lam,lr):
    return next(r for r in fac if (r['mask'],r['lambda'],int(r['lr_multiplier']))==(mask,lam,lr))
d1=cell('Dense','1.0',1); d20=cell('Dense','1.0',20)
dl20=cell('Dense','0.1',20); m20=cell('Top-0.4','1.0',20); r20=cell('Top-0.4','0.1',20)
close('recipe20-dense1',float(r20['mean'])-float(d1['mean']),.019)
close('recipe20-dense20',float(r20['mean'])-float(d20['mean']),.105)
close('recipe20-dense-lambda20',float(r20['mean'])-float(dl20['mean']),.045)
close('recipe20-mask-only20',float(r20['mean'])-float(m20['mean']),.035)
close('20x interaction',float(r20['mean'])-float(m20['mean'])-float(dl20['mean'])+float(d20['mean']),-.025)
diff=[float(r20[f'seed{s}'])-float(d1[f'seed{s}']) for s in (42,43,44)]
mean,sd=st.mean(diff),st.stdev(diff); se=sd/math.sqrt(3); margin=4.303*se
close('paired mean',mean,.019); close('paired sd',sd,.002645751,1e-8)
close('CI lower',mean-margin,.012427059,1e-8); close('CI upper',mean+margin,.025572941,1e-8)

lr=list(csv.DictReader(open('audited_lr_audit.csv',encoding='utf-8')))
if len(lr)!=4: fail.append('LR audit does not have four rows')
for r in lr:
    vals=[float(r[k]) for k in ('requested_lr','optimizer_lr','scheduler_lr','logged_actor_lr')]
    if len(set(vals))!=1 or int(r['mismatch_count'])!=0: fail.append(f"LR mismatch: {r}")

sp=list(csv.DictReader(open('audited_spine_results.csv',encoding='utf-8')))
for key,want_mu,want_sd in [('jaccard',.600,.009643651),('recall',.749666667,.007571878),
                            ('sign_agreement',.780,.005196152),('enrichment',2.400,.038574603)]:
    vals=[float(r[key]) for r in sp]
    close(f'{key} mean',st.mean(vals),want_mu)
    close(f'{key} sd',st.stdev(vals),want_sd)
close('random Jaccard',.4/(2-.4),.25)
close('retained norm',math.sqrt(.85),.922,5e-4)

# M1 structured null: validate the complete 3x3 Spine--dense matrix, all three
# dense--dense pairs, and the row-wise matched specificity contrast.
spec=list(csv.DictReader(open('audited_specificity_results.csv',encoding='utf-8')))
cross=[r for r in spec if r['comparison']=='spine_dense']
dd=[r for r in spec if r['comparison']=='dense_dense']
rand=[r for r in spec if r['comparison']=='random']
if len(cross)!=9 or len(dd)!=3 or len(rand)!=1:
    fail.append(f'specificity design rows: cross={len(cross)}, dense-dense={len(dd)}, random={len(rand)}')
expected_cross={(42,42):.604,(42,43):.418,(42,44):.409,
                (43,42):.431,(43,43):.589,(43,44):.414,
                (44,42):.427,(44,43):.423,(44,44):.607}
observed_cross={(int(r['left_seed']),int(r['right_seed'])):float(r['jaccard']) for r in cross}
if set(observed_cross)!=set(expected_cross): fail.append('specificity cross matrix is incomplete or duplicated')
for pair,want in expected_cross.items(): close(f'specificity cross {pair}',observed_cross.get(pair,float('nan')),want)
expected_dd={(42,43):.414,(42,44):.423,(43,44):.419}
observed_dd={(int(r['left_seed']),int(r['right_seed'])):float(r['jaccard']) for r in dd}
if set(observed_dd)!=set(expected_dd): fail.append('dense-dense pairs are incomplete or duplicated')
for pair,want in expected_dd.items(): close(f'dense-dense {pair}',observed_dd.get(pair,float('nan')),want)
if rand: close('specificity random row',float(rand[0]['jaccard']),.250)
matched=[observed_cross[(s,s)] for s in (42,43,44)]
mismatched=[v for (s,d),v in observed_cross.items() if s!=d]
dense_dense=list(observed_dd.values())
specificity=[observed_cross[(s,s)]-st.mean(observed_cross[(s,d)] for d in (42,43,44) if d!=s)
             for s in (42,43,44)]
close('matched mean',st.mean(matched),.600); close('matched sd',st.stdev(matched),.009643651,1e-8)
close('mismatched mean',st.mean(mismatched),.420333333,1e-8); close('mismatched sd',st.stdev(mismatched),.008238123,1e-8)
close('dense-dense mean',st.mean(dense_dense),.418666667,1e-8); close('dense-dense sd',st.stdev(dense_dense),.004509250,1e-8)
for s,got,want in zip((42,43,44),specificity,(.1905,.1665,.182)): close(f'specificity gain seed{s}',got,want)
close('specificity gain mean',st.mean(specificity),.179666667,1e-8)
close('specificity gain sd',st.stdev(specificity),.012168950,1e-8)
close('normalized specificity',(st.mean(matched)-st.mean(dense_dense))/(1-st.mean(dense_dense)),.311926606,1e-8)
spine_by_seed={int(r['seed']):float(r['jaccard']) for r in sp}
for s in (42,43,44):
    close(f'spine/specificity matched identity seed{s}',observed_cross[(s,s)],spine_by_seed[s])
if min(matched)<=max(mismatched+dense_dense):
    fail.append('matched diagonal does not strictly exceed every structured-null value')

controls=list(csv.DictReader(open('audited_mask_controls.csv',encoding='utf-8')))
transfer=list(csv.DictReader(open('audited_transfer_results.csv',encoding='utf-8')))
stability=list(csv.DictReader(open('audited_stability_results.csv',encoding='utf-8')))
q17=list(csv.DictReader(open('audited_qwen17b_results.csv',encoding='utf-8')))
primary=list(csv.DictReader(open('audited_primary_benchmark_results.csv',encoding='utf-8')))
if len(controls)!=6: fail.append('mask controls do not have six rows')
if len(transfer)!=4: fail.append('transfer table does not have four rows')
if len(stability)!=2: fail.append('stability table does not have two rows')
if len(q17)!=6: fail.append('Qwen3-1.7B table does not have six rows')
if len(primary)!=2: fail.append('primary benchmark decomposition does not have two rows')
for r in transfer: close(f"{r['setting']} gain",float(r['recipe_mean4'])-float(r['dense_mean4']),float(r['gain']))

# Primary Qwen3-8B decomposition. Benchmark columns and Mean4 are independently
# rounded three-seed aggregates, so validate the reported values and direction
# without reconstructing Mean4 from rounded components.
primary_index={r['condition']:r for r in primary}
expected_primary={
    'Dense 1x':(.762,.227,.200,.513,.426),
    'Spine recipe 20x':(.800,.240,.210,.530,.445),
}
if set(primary_index)!=set(expected_primary):
    fail.append('primary benchmark conditions are incomplete or duplicated')
for condition,wants in expected_primary.items():
    r=primary_index.get(condition)
    if r is None: continue
    if r['seeds']!='42;43;44' or r['aggregation']!='three-seed aggregate':
        fail.append(f'{condition} primary benchmark provenance mismatch')
    for key,want in zip(('math500','aime24','aime25','olympiad','mean4'),wants):
        close(f'{condition} primary {key}',float(r[key]),want)
if set(primary_index)==set(expected_primary):
    dense_primary=primary_index['Dense 1x']
    recipe_primary=primary_index['Spine recipe 20x']
    for key in ('math500','aime24','aime25','olympiad'):
        if float(recipe_primary[key])<=float(dense_primary[key]):
            fail.append(f'primary benchmark gain is not positive for {key}')
    close('primary Dense Mean4/factorial identity',float(dense_primary['mean4']),float(d1['mean']))
    close('primary Recipe Mean4/factorial identity',float(recipe_primary['mean4']),float(r20['mean']))

# M4: per-seed Qwen3-1.7B decomposition and the non-AIME Mean2 sensitivity.
q17_index={(r['condition'],int(r['seed'])):r for r in q17}
if set(q17_index)!={(c,s) for c in ('Dense','Recipe') for s in (42,43,44)}:
    fail.append('Qwen3-1.7B condition/seed design is incomplete or duplicated')
for (condition,seed),r in q17_index.items():
    for key,total in [('math500',16000),('aime24',960),('aime25',960),('olympiad',18592)]:
        correct=int(r[f'{key}_correct'])
        observed_total=int(r[f'{key}_total'])
        if observed_total!=total: fail.append(f'Qwen3-1.7B {condition} seed{seed} {key} total mismatch')
        close(f'Qwen3-1.7B {condition} seed{seed} {key} count ratio',float(r[key]),correct/observed_total,1e-8)
    vals=[float(r[k]) for k in ('math500','aime24','aime25','olympiad')]
    close(f'Qwen3-1.7B {condition} seed{seed} Mean4',float(r['mean4']),st.mean(vals))
    close(f'Qwen3-1.7B {condition} seed{seed} Mean2',float(r['mean2']),st.mean((vals[0],vals[3])))
q17_summary={}
for condition in ('Dense','Recipe'):
    rows=[q17_index[(condition,s)] for s in (42,43,44)]
    for key,want_mu,want_sd in [
        ('math500',.74200000 if condition=='Dense' else .74066667,
                    .00556776 if condition=='Dense' else .00802081),
        ('aime24',.11944444 if condition=='Dense' else .15069444,
                   .00809110 if condition=='Dense' else .00898091),
        ('aime25',.13194444 if condition=='Dense' else .14861111,
                   .00693576 if condition=='Dense' else .00739021),
        ('olympiad',.46313827 if condition=='Dense' else .48149742,
                    .00627008 if condition=='Dense' else .00625036),
        ('mean4',.36413179 if condition=='Dense' else .38036741,
                 .00645540 if condition=='Dense' else .00760270),
        ('mean2',.60256913 if condition=='Dense' else .61108204,
                 .00591653 if condition=='Dense' else .00713545)]:
        vals=[float(r[key]) for r in rows]
        close(f'Qwen3-1.7B {condition} {key} mean',st.mean(vals),want_mu)
        close(f'Qwen3-1.7B {condition} {key} sd',st.stdev(vals),want_sd)
        q17_summary[(condition,key)]=st.mean(vals)
mean2_gain=[float(q17_index[('Recipe',s)]['mean2'])-float(q17_index[('Dense',s)]['mean2']) for s in (42,43,44)]
for s,got,want in zip((42,43,44),mean2_gain,(.00796644,.00252409,.01504819)):
    close(f'Qwen3-1.7B seed{s} Mean2 gain',got,want,1e-8)
if sum(x>0 for x in mean2_gain)!=3: fail.append('Qwen3-1.7B Recipe does not win Mean2 in all three seeds')
close('Qwen3-1.7B paired Mean2 gain SD',st.stdev(mean2_gain),.00627991,1e-8)
mean4_gain=[float(q17_index[('Recipe',s)]['mean4'])-float(q17_index[('Dense',s)]['mean4']) for s in (42,43,44)]
for s,got,want in zip((42,43,44),mean4_gain,(.01622280,.01141829,.02106576)):
    close(f'Qwen3-1.7B seed{s} Mean4 gain',got,want,1e-8)
close('Qwen3-1.7B paired Mean4 gain SD',st.stdev(mean4_gain),.00482375,1e-8)
if [float(q17_index[('Dense',s)]['mean4']) for s in (42,43,44)].index(max(float(q17_index[('Dense',s)]['mean4']) for s in (42,43,44))) != 1:
    fail.append('Qwen3-1.7B Dense seed ordering no longer matches count audit')
if [float(q17_index[('Recipe',s)]['mean4']) for s in (42,43,44)].index(max(float(q17_index[('Recipe',s)]['mean4']) for s in (42,43,44))) != 2:
    fail.append('Qwen3-1.7B Recipe seed ordering no longer matches count audit')
for key,want in [('math500',-.00133333),('aime24',.03125),('aime25',.01666667),
                 ('olympiad',.01835915),('mean4',.01623562),('mean2',.00851291)]:
    close(f'Qwen3-1.7B paired {key} gain',q17_summary[('Recipe',key)]-q17_summary[('Dense',key)],want)
q17_transfer=next((r for r in transfer if r['setting']=='Qwen3-1.7B'),None)
if q17_transfer is None:
    fail.append('transfer summary lacks Qwen3-1.7B')
else:
    close('Qwen3-1.7B transfer dense',float(q17_transfer['dense_mean4']),q17_summary[('Dense','mean4')])
    close('Qwen3-1.7B transfer recipe',float(q17_transfer['recipe_mean4']),q17_summary[('Recipe','mean4')])
stab={r['condition']:r for r in stability}
for condition,reward,kl,entropy,clip,degraded in [
    ('Spine recipe 20x',.080,.030,.250,.100,0),
    ('Dense 20x',-.150,.090,.150,.650,1)]:
    r=stab.get(condition)
    if r is None: fail.append(f'missing stability row {condition}'); continue
    for key,want in [('final_reward',reward),('kl_loss',kl),('entropy',entropy),
                     ('response_clip_ratio',clip)]: close(f'{condition} {key}',float(r[key]),want)
    for field in ('run_failure','numerical_collapse','policy_collapse'):
        if int(r[field])!=0: fail.append(f'{condition} unexpected {field}')
    if int(r['severe_performance_degradation'])!=degraded:
        fail.append(f'{condition} severe-degradation mismatch')

# Systems reporting is recomputed from the normalized cross-cluster export.
env=[json.loads(line) for line in open('audited_remote_export/environment_manifest.jsonl',encoding='utf-8') if line.strip()]
if not env or any(r['gpu_model']!='NVIDIA H200' or int(r['gpu_count'])!=8 for r in env):
    fail.append('environment manifest is not uniformly 8x NVIDIA H200')
timing=list(csv.DictReader(open('audited_remote_export/efficiency_step_timings.csv',encoding='utf-8')))
memory=list(csv.DictReader(open('audited_remote_export/efficiency_peak_memory.csv',encoding='utf-8')))
formal=[json.loads(line) for line in open('audited_remote_export/formal_run_manifests.jsonl',encoding='utf-8') if line.strip()]
for condition,want in [('dense',12.006804),('recipe',13.207484)]:
    values=[float(r['step_wall_clock_seconds']) for r in timing if r['condition']==condition]
    close(f'{condition} median step time',st.median(values),want,1e-6)
for condition,want in [('dense',72.252000),('recipe',75.864600)]:
    values=[float(r['peak_allocated_memory_bytes'])/2**30 for r in memory if r['condition']==condition]
    close(f'{condition} mean peak memory GiB',st.mean(values),want,1e-6)
node_hours=sum(float(r['wall_clock_seconds']) for r in formal)/3600
close('formal end-to-end node hours',node_hours,1332.65,1e-6)
close('formal end-to-end GPU hours',node_hours*8,10661.2,1e-6)

tex=open('main_ne.tex',encoding='utf-8').read()
tex_flat=' '.join(tex.split())
highlights=open('highlights.txt',encoding='utf-8').read()
cover=open('cover_letter.txt',encoding='utf-8').read()
concept=open('make_fig_concept.py',encoding='utf-8').read()
contam=open('make_fig_contam.py',encoding='utf-8').read()
licenses=open('LICENSE_AND_DATA_USE.md',encoding='utf-8').read()
ai_declaration=open('declarations/generative_ai.txt',encoding='utf-8').read()
submission_metadata=open('submission_metadata.txt',encoding='utf-8').read()
citation_metadata=open('CITATION.cff',encoding='utf-8').read()
required=['0.445','0.340','0.105','0.600\\pm0.010','0.750\\pm0.008',
          '0.780\\pm0.005','0.420\\pm0.008','0.419\\pm0.005','0.180\\pm0.012',
          '0.36413\\pm0.00646','0.38037\\pm0.00760','0.60257\\pm0.00592',
          '0.61108\\pm0.00714','0.01624\\pm0.00482','0.00851\\pm0.00628',
          '[0.0124,0.0256]']
for token in required:
    if token not in tex: fail.append(f'manuscript missing canonical token {token}')
for artifact in ['audited_factorial_results.csv','audited_stability_results.csv',
                 'audited_specificity_results.csv','audited_qwen17b_results.csv',
                 'audited_channel_overlap_results.csv',
                 'audited_primary_benchmark_results.csv']:
    if not Path(artifact).is_file(): fail.append(f'evidence bundle missing {artifact}')
for token in ['Lazy Likelihood Displacement','negative-sample reinforcement',
              'step-size--conditional','+0.002','+0.060',
              'do not infer that negative samples should generally']:
    if token not in tex: fail.append(f'manuscript missing M5 conditionality token {token}')
for token in ['avg@32','temperature 0.6','30\\times32=960','16,000','18,592',
              'RL step 0','steps 40, 80, 120, and 160',
              'severe performance degradation','not an RL resume']:
    if token not in tex_flat: fail.append(f'manuscript missing protocol/status token {token}')
for token in ['12.007','13.207','72.25','75.86','10,661.2 H200-GPU-hours',
              'includes evaluation and','over 160 RL steps',
              'lightly tuned dense',
              'recovers Mean4 from 0.340 to 0.383',
              'gap $0.062$',
              'not universal superiority over every retuned',
              'not driven solely by the two smaller AIME sets',
              'MATH500 and AIME24 cards do not declare a license',
              'LICENSE\\_AND\\_DATA\\_USE.md']:
    if token not in tex_flat: fail.append(f'manuscript missing scope/license token {token}')
tuned=list(csv.DictReader(open('audited_tuned_dense_results.csv',encoding='utf-8')))
tuned_map={r['condition']:float(r['mean4']) for r in tuned}
for cond,want in [('Untuned Dense 20x',.340),('Tuned Dense 20x',.383),
                  ('Dense 1x',.426),('Recipe 20x',.445)]:
    close(f'tuned-dense table {cond}',tuned_map.get(cond,float('nan')),want)
close('tuned-dense recipe-tuned gap',tuned_map['Recipe 20x']-tuned_map['Tuned Dense 20x'],.062)

# Figure 2 channel diagnostics: validate every seed record, recompute mean and
# sample SD, and keep the two enrichment estimands explicitly separate.
channel_rows=list(csv.DictReader(open('audited_channel_overlap_results.csv',encoding='utf-8')))
channel_seed={(r['metric'],int(r['seed'])):r for r in channel_rows if r['record_type']=='seed'}
expected_channel_seed={
    'positive_active_fraction':(.049,.046,.049),
    'negative_active_fraction':(.047,.049,.048),
    'measured_overlap_fraction':(.0164,.0175,.0171),
    'same_sign_rate':(.491,.472,.477),
    'conflict_rate':(.509,.528,.523),
    'positive_gini':(.691,.713,.696),
    'negative_gini':(.622,.638,.630),
    'positive_top1_energy':(.869,.891,.880),
    'negative_top1_energy':(.831,.848,.841),
}
expected_seed_keys={(metric,seed) for metric in expected_channel_seed for seed in (42,43,44)}
expected_seed_keys|={(metric,seed) for metric in ('independent_overlap_reference','overlap_enrichment')
                    for seed in (42,43,44)}
if set(channel_seed)!=expected_seed_keys:
    fail.append(f'Figure 2 seed design mismatch: {sorted(set(channel_seed)^expected_seed_keys)}')
for metric,wants in expected_channel_seed.items():
    for seed,want in zip((42,43,44),wants):
        row=channel_seed.get((metric,seed))
        if row is None: continue
        close(f'Figure 2 {metric} seed{seed}',float(row['value']),want,1e-12)
        expected_status='derived' if metric=='conflict_rate' else 'observed'
        if row['status']!=expected_status:
            fail.append(f'Figure 2 {metric} seed{seed} status mismatch')

channel_summary={(r['metric'],r['record_type']):r for r in channel_rows
                 if r['record_type']!='seed'}
summary_metrics=tuple(expected_channel_seed)+('independent_overlap_reference','overlap_enrichment')
for metric in summary_metrics:
    values=[float(channel_seed[(metric,seed)]['value']) for seed in (42,43,44)]
    for record_type,want in [('mean',st.mean(values)),('sample_sd',st.stdev(values))]:
        row=channel_summary.get((metric,record_type))
        if row is None:
            fail.append(f'Figure 2 missing {metric} {record_type}')
        else:
            close(f'Figure 2 {metric} {record_type}',float(row['value']),want,1e-12)
            if row['status']!='derived': fail.append(f'Figure 2 {metric} {record_type} is not derived')

for seed in (42,43,44):
    positive=float(channel_seed[('positive_active_fraction',seed)]['value'])
    negative=float(channel_seed[('negative_active_fraction',seed)]['value'])
    overlap=float(channel_seed[('measured_overlap_fraction',seed)]['value'])
    independent=float(channel_seed[('independent_overlap_reference',seed)]['value'])
    enrichment=float(channel_seed[('overlap_enrichment',seed)]['value'])
    same=float(channel_seed[('same_sign_rate',seed)]['value'])
    conflict=float(channel_seed[('conflict_rate',seed)]['value'])
    close(f'Figure 2 independent reference seed{seed}',independent,positive*negative,1e-12)
    close(f'Figure 2 enrichment seed{seed}',enrichment,overlap/independent,1e-12)
    close(f'Figure 2 sign partition seed{seed}',same+conflict,1.0,1e-12)

aggregate_reference=channel_summary.get(('independent_overlap_reference','aggregate_product'))
aggregate_enrichment=channel_summary.get(('overlap_enrichment','ratio_of_aggregate_quantities'))
if aggregate_reference is None or aggregate_enrichment is None:
    fail.append('Figure 2 lacks explicit ratio-of-aggregate records')
else:
    mean_value=lambda metric: float(channel_summary[(metric,'mean')]['value'])
    aggregate_reference_value=float(aggregate_reference['value'])
    aggregate_enrichment_value=float(aggregate_enrichment['value'])
    close('Figure 2 aggregate-product reference',aggregate_reference_value,
          mean_value('positive_active_fraction')*mean_value('negative_active_fraction'),1e-12)
    close('Figure 2 ratio of aggregate quantities',aggregate_enrichment_value,
          mean_value('measured_overlap_fraction')/aggregate_reference_value,1e-12)
    close('Figure 2 mean-of-ratios estimand',mean_value('overlap_enrichment'),7.385176549806,1e-12)
    if abs(aggregate_enrichment_value-mean_value('overlap_enrichment'))<1e-6:
        fail.append('Figure 2 enrichment estimands were accidentally conflated')

for row in channel_rows:
    value=float(row['value'])
    if row['unit'] in ('fraction','index') and not 0<=value<=1:
        fail.append(f'Figure 2 {row["metric"]}/{row["record_type"]} is outside [0,1]')
for token in ['4.8\\%','1.7\\%','0.048^2=0.23\\%','7.4\\times',
              '48\\% of signs agree and 52\\% conflict','Gini 0.70 and 0.63',
              'top-1\\% energy','0.88 and 0.84']:
    if token not in tex: fail.append(f'manuscript missing Figure 2 token {token}')
if 'audited_channel_overlap_results.csv' not in contam:
    fail.append('Figure 2 generator does not read the channel audit CSV')

for token in ['Qwen3-8B and Qwen3-1.7B','Apache License 2.0',
              'Llama 3.1 Community License','MATH500','does not declare a license',
              'AIME 2024','AIME 2025','OlympiadBench','MIT License','VERL']:
    if token not in licenses: fail.append(f'license matrix missing canonical token {token}')

# The opening figure is explicitly schematic, but every numerical callout must
# remain tied to the audited tables.  Read constants from its generator rather
# than trusting rendered labels.
def concept_constant(name):
    m=re.search(rf'^\s*{name}\s*=\s*(-?\d+(?:\.\d+)?)\s*$',concept,re.M)
    if not m:
        fail.append(f'concept figure missing constant {name}')
        return float('nan')
    return float(m.group(1))

close('concept DENSE20_MEAN4',concept_constant('DENSE20_MEAN4'),float(d20['mean']))
close('concept SPINE20_MEAN4',concept_constant('SPINE20_MEAN4'),float(r20['mean']))
close('concept GAIN_VS_DENSE20',concept_constant('GAIN_VS_DENSE20'),
      float(r20['mean'])-float(d20['mean']))
close('concept NONZERO_RATIO',concept_constant('NONZERO_RATIO'),.393)
close('concept ENERGY_KEPT',concept_constant('ENERGY_KEPT'),.850)
close('concept JACCARD_MEAN',concept_constant('JACCARD_MEAN'),
      st.mean(float(r['jaccard']) for r in sp))
close('concept JACCARD_SD',concept_constant('JACCARD_SD'),
      st.stdev(float(r['jaccard']) for r in sp))
if 'SCHEMATIC' not in concept or 'schematic' not in tex.lower():
    fail.append('concept schematic disclosure missing from source/caption: schematic')
if 'not measured coordinates' not in concept or 'not measured coordinates' not in tex:
    fail.append('concept schematic disclosure missing from source/caption: not measured coordinates')
if 'collapse: yes' in concept:
    fail.append('concept figure retains obsolete collapse label')
if 'severe degradation' not in concept:
    fail.append('concept figure lacks audited severe-degradation label')

# Submission identity and AI disclosure must agree across the manuscript and
# paste-ready supporting files. Unknown phone/postcode fields remain explicit
# author-input items rather than inferred metadata.
for token in ['Shikai Li','Qingsong Cai','Rui Shi','lishikai@wchscu.edu.cn',
              'dr.shirui@hotmail.com','West China Hospital, Sichuan University']:
    if token not in tex: fail.append(f'manuscript missing submission identity token {token}')
    if token not in submission_metadata: fail.append(f'submission metadata missing identity token {token}')
for token in ['given-names: Shikai','given-names: Qingsong','given-names: Rui',
              'lishikai@wchscu.edu.cn','dr.shirui@hotmail.com']:
    if token not in citation_metadata: fail.append(f'CITATION.cff missing identity token {token}')
for token in ['Cursor and OpenAI Codex','synthesize image',
              'produced deterministically from author-verified data',
              'all AI-assisted code was reviewed and executed under author control']:
    if token not in tex_flat: fail.append(f'manuscript AI disclosure missing token {token}')
    if token not in ai_declaration: fail.append(f'standalone AI disclosure missing token {token}')
for obsolete in ['fabricate data, numerical results, tables, or scientific figures',
                 'No generative AI system was used to create or alter']:
    if obsolete in tex or obsolete in ai_declaration:
        fail.append(f'AI disclosure retains ambiguous obsolete wording: {obsolete}')

# The manuscript's full factorial table must reproduce all 48 seed values and
# all 16 mean/SD pairs in CSV order. This catches a correct CSV paired with a
# silently mistyped table cell.
tm=__import__('re').search(r'\\label\{tab:factorial\}(.*?)\\end\{tabular\}',tex,__import__('re').S)
if not tm:
    fail.append('manuscript factorial table missing')
else:
    body=__import__('re').sub(r'\\textbf\{([^{}]*)\}',r'\1',tm.group(1))
    printed=[]
    for line in body.splitlines():
        if '&' not in line or '$\\pm$' not in line: continue
        cells=line.split('&')[-4:]
        nums=[]
        for c in cells: nums.extend(float(x) for x in __import__('re').findall(r'-?(?:0?\.)?\d+',c))
        if len(nums)==5: printed.append(nums)
    if len(printed)!=16:
        fail.append(f'manuscript factorial table has {len(printed)} parsed rows, expected 16')
    else:
        for r,g in zip(fac,printed):
            want=[float(r[f'seed{s}']) for s in (42,43,44)]+[float(r['mean']),float(r['sample_std'])]
            for j,(got,exp) in enumerate(zip(g,want)):
                # Mean4 cells are printed to three decimals and their SDs to
                # four; the CSV retains enough precision for recomputation.
                close(f"manuscript factorial row {r['mask']}/{r['lambda']}/{r['lr_multiplier']} col{j}",
                      got,exp,5e-5 if j==4 else 5e-7)

# The compact Qwen3-1.7B manuscript table must reproduce every aggregate and
# component contrast, not merely mention the Mean2 headline in prose.  The
# publication layout may split the four benchmark columns and two summary
# columns into stacked tabular blocks, so concatenate every matching row within
# the labelled table rather than assuming one wide row.
qm=re.search(r'\\label\{tab:qwen17\}(.*?)\\end\{table\}',tex,re.S)
if not qm:
    fail.append('manuscript Qwen3-1.7B table missing')
else:
    qbody=qm.group(1)
    for label,want in [
        ('Dense',[.742,.006,.119,.008,.132,.007,.463,.006,.36413,.00646,.60257,.00592]),
        ('Recipe',[.741,.008,.151,.009,.149,.007,.481,.006,.38037,.00760,.61108,.00714]),
        ('Delta',[-.001,.031,.017,.018,.01624,.00851])]:
        pattern=r'^' + (r'\$\\Delta\$' if label=='Delta' else label) + r'\s*&([^\n]+)'
        matches=re.findall(pattern,qbody,re.M)
        if not matches:
            fail.append(f'manuscript Qwen3-1.7B row missing: {label}')
            continue
        got=[]
        for row in matches:
            got.extend(float(x) for x in re.findall(r'[+-]?(?:\d+\.\d+|\.\d+)',row))
        if len(got)!=len(want):
            fail.append(f'manuscript Qwen3-1.7B {label} has {len(got)} values, expected {len(want)}')
            continue
        for j,(g,w) in enumerate(zip(got,want)): close(f'manuscript Qwen3-1.7B {label} col{j}',g,w)

# Recompute the four contrasts printed in tab:effects.
for lr,mask_eff,down_eff,joint,interaction in [
    (1,-.005,.002,.000,.003),(5,.000,.015,.014,-.001),
    (10,.018,.015,.045,.012),(20,.070,.060,.105,-.025)]:
    d=float(cell('Dense','1.0',lr)['mean']); dl=float(cell('Dense','0.1',lr)['mean'])
    m=float(cell('Top-0.4','1.0',lr)['mean']); rr=float(cell('Top-0.4','0.1',lr)['mean'])
    close(f'{lr}x mask effect',m-d,mask_eff); close(f'{lr}x downweight effect',dl-d,down_eff)
    close(f'{lr}x joint gain',rr-d,joint); close(f'{lr}x interaction',rr-m-dl+d,interaction)
for banned in ['0.650\\pm0.012','88\\% of update energy','collapses at $10\\times$']:
    if banned in tex: fail.append(f'manuscript retains superseded claim {banned}')

file_bullets=[' '.join(x[2:].split()) for x in highlights.splitlines() if x.startswith('- ')]
if not 3<=len(file_bullets)<=5: fail.append(f'highlights count {len(file_bullets)} is outside 3--5')
for i,b in enumerate(file_bullets,1):
    if len(b)>85: fail.append(f'highlight {i} has {len(b)} characters (>85)')
for token in ['16-cell','0.086','0.019','[0.0124, 0.0256]','0.181']:
    if token not in highlights: fail.append(f'highlights missing canonical token {token}')
for token in ['0.426','0.340','0.445','0.019','[0.0124, 0.0256]',
              '0.600±0.010','0.420±0.008','0.419±0.005','0.180±0.012']:
    if token not in cover: fail.append(f'cover letter missing canonical token {token}')

if fail:
    print('FAILED:'); print('\n'.join(' - '+x for x in fail)); sys.exit(1)
print('OK: factorial, LR audit, paired CI, M1 specificity, M4 Mean2, Figure 2, Spine and transfer checks passed')
