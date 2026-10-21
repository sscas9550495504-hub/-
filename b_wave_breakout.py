#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台股 B 波突破 -> C 波 -> 回升趨勢的全市場研究工具。
信號只用當時與過去日K；下一交易日開盤模擬成交。
結果屬形態事件研究，不是保證獲利或 Elliott 波浪的真實標註。
"""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from market_abc_core import get_data, enrich, COST_BUY, COST_SELL

HORIZONS = (10, 20, 40, 60)
CATEGORIES = {
    'WATCH_B':'B波整理／監控', 'NEAR_B':'接近B波上緣',
    'B_BREAK':'B波剛突破／C初段', 'C_RUN':'B突破後C波進行中',
    'A_BREAK':'突破A高／續攻', 'A_PULLBACK':'突破A高後回測', 'FAILED':'B突破失敗',
    'A_FORM':'A波形成中', 'NO_PATTERN':'尚無匹配型態',
    'STALE':'個股行情過期'
}
FEATURE_NAMES = {'b_retrace':'B回撤','vol_ratio':'量比','rsi':'RSI14',
                 'macd_hist':'MACD histogram','b_vol_contraction':'B量縮程度'}


def _f(value):
    try:
        v=float(value)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):return np.nan


def _event(code, name, d, i, state, kind, ph, p0, a_hi, b_lo, b_start, a_start,
           r, breakdown, prior20_turnover, bottom_i):
    return dict(symbol=code,name=name,signal_type=kind,signal_date=d.iloc[i].date.date().isoformat(),
        signal_index=i,crash_peak=ph,bottom=p0,a_high=a_hi,b_low=b_lo,
        crash_pct=100*(p0/ph-1),a_rebound_pct=100*(a_hi/p0-1),
        b_retrace=(a_hi-b_lo)/(a_hi-p0) if a_hi>p0 else np.nan,
        b_days=i-b_start,days_from_bottom=i-bottom_i,breakout_ref=r,
        breakout_pct=100*(d.iloc[i].close/r-1) if np.isfinite(r) and r else np.nan,
        rsi=_f(d.iloc[i].rsi),vol_ratio=_f(d.iloc[i].vol_ratio),
        macd_hist=_f(d.iloc[i].macd_hist),
        macd_rising=bool(d.iloc[i].hist_rising),
        ma20_slope5=_f(d.iloc[i].ma20 / d.iloc[i-5].ma20-1) if i>=5 else np.nan,
        b_vol_contraction=_f(d.iloc[i-4:i+1].volume.mean()/d.iloc[i-19:i+1].volume.mean()) if i>=20 else np.nan,
        avg_turnover20=_f(prior20_turnover),
        bottom_date=d.iloc[bottom_i].date.date().isoformat(),
        a_start_date=d.iloc[a_start].date.date().isoformat())


def inspect_symbol(code, df, summer_from, summer_to, turnover_limit):
    """增量狀態機，僅從過去行情判斷波段；不能用右側高低點確定歷史進場信號。"""
    d=enrich(df.sort_values('date')).reset_index(drop=True)
    d['avg_turnover20']=(d['close']*d['volume']).shift(1).rolling(20).mean()
    d['ma60']=d['close'].rolling(60).mean()
    n=len(d)
    if n<105: return None,[],d
    state='SEARCH'; ph=p0=a_hi=b_lo=np.nan
    peak_i=bottom_i=a_i=b_start=b_low_i=None
    signal_i=None; a_break_i=None; b_res=np.nan
    events=[]
    date=d.date.to_numpy(); hi=d.high.to_numpy(); lo=d.low.to_numpy();close=d.close.to_numpy()
    vol=d.volume.to_numpy(); rsi=d.rsi.to_numpy();vr=d.vol_ratio.to_numpy();mh=d.macd_hist.to_numpy()
    ma20=d.ma20.to_numpy();avgturn=d.avg_turnover20.to_numpy()
    def reset():
        nonlocal state,ph,p0,a_hi,b_lo,peak_i,bottom_i,a_i,b_start,b_low_i,signal_i,a_break_i,b_res
        state='SEARCH'; ph=p0=a_hi=b_lo=b_res=np.nan
        peak_i=bottom_i=a_i=b_start=b_low_i=signal_i=a_break_i=None
    for i in range(65,n):
        if vol[i]<=0:continue
        if state=='SEARCH':
            # 價格前65日高點下降至少30%，高點須出現於最近55根K線
            s=max(0,i-65);pp=int(np.argmax(hi[s:i]))+s
            peak=hi[pp]
            if peak>0 and close[i]/peak<=.70 and i-pp<=55:
                ph=peak;peak_i=pp;p0=lo[i];bottom_i=i;state='BOTTOM'
        elif state=='BOTTOM':
            if lo[i]<p0:p0=lo[i];bottom_i=i
            if i-bottom_i>45 or i-peak_i>120:reset();continue
            if close[i]>=p0*1.18 and i-bottom_i>=2:
                a_hi=hi[i];a_i=i;state='A'
        elif state=='A':
            if hi[i]>a_hi:a_hi=hi[i];a_i=i
            if a_hi>=p0*1.25 and i-a_i>=3 and lo[i] <= a_hi-.25*(a_hi-p0):
                b_lo=lo[i];b_low_i=i;b_start=i;state='B'
            if i-bottom_i>120:reset()
        elif state=='B':
            if lo[i]<b_lo:b_lo=lo[i];b_low_i=i
            depth=(a_hi-b_lo)/(a_hi-p0)
            if b_lo<=p0*1.005 or depth>=.94 or i-bottom_i>135:reset();continue
            # B波動態上緣：排除今日的前12根K棒高點；至少5根B K線
            ref_hi=np.max(hi[max(a_i+1,i-12):i]) if i-max(a_i+1,i-12)>=3 else np.nan
            b_res=min(ref_hi,a_hi) if np.isfinite(ref_hi) else np.nan
            if i-b_start<5 or not np.isfinite(b_res):continue
            broke=close[i]>b_res*1.005 and close[i-1]<=b_res*1.005
            momentum=(np.isfinite(rsi[i]) and 46<=rsi[i]<=76 and
                np.isfinite(vr[i]) and vr[i]>=1.20 and
                np.isfinite(mh[i]) and mh[i]>mh[i-1] and
                close[i]>ma20[i])
            # 首次B波上緣突破；如果同時突破A高，單獨標示為直接A高突破
            if broke and momentum and np.isfinite(avgturn[i]) and avgturn[i]>=turnover_limit:
                kind='B_上緣突破' if close[i]<a_hi*1.01 else 'A_高點直接突破'
                events.append(_event(code,'',d,i,state,kind,ph,p0,a_hi,b_lo,b_start,a_i,b_res,
                                     None,avgturn[i],bottom_i))
                signal_i=i
                if close[i]>=a_hi*1.01:
                    a_break_i=i;state='ABOVE_A'
                else:state='C'
            elif close[i]>a_hi*1.01 and close[i-1]<=a_hi*1.01:
                # 未具備B量價條件時，A高突破只列觀察，不當成B買進訊號
                a_break_i=i;state='ABOVE_A'
        elif state in ('C','ABOVE_A'):
            if close[i] < b_lo*.995 or lo[i]<=p0 or i-bottom_i>180:
                state='FAILED';continue
            if state=='C' and close[i]>=a_hi*1.01 and close[i-1]<a_hi*1.01:
                a_break_i=i;state='ABOVE_A'
        elif state=='FAILED':
            if i-bottom_i>185 or lo[i]<p0:reset()
    last=d.iloc[-1];i=n-1
    prior20=_f(last.avg_turnover20)
    recent=bool(bottom_i is not None and summer_from<=date[bottom_i]<=summer_to)
    stage='NO_PATTERN';near_gap=np.nan;score=0
    if recent and state!='SEARCH' and np.isfinite(p0):
        depth=(a_hi-b_lo)/(a_hi-p0) if np.isfinite(b_lo) and a_hi>p0 else np.nan
        if state=='A':stage='A_FORM';score=25
        elif state=='B':
            ref_hi=np.max(hi[max(a_i+1,i-12):i]) if i-max(a_i+1,i-12)>=3 else np.nan
            b_res=min(ref_hi,a_hi) if np.isfinite(ref_hi) else np.nan
            near_gap=100*(b_res/close[i]-1) if np.isfinite(b_res) else np.nan
            stage='NEAR_B' if np.isfinite(near_gap) and -1<=near_gap<=4 else 'WATCH_B'
            score=56 if stage=='NEAR_B' else 38
        elif state=='C':stage='B_BREAK' if i-signal_i<=5 else 'C_RUN';score=75 if i-signal_i<=5 else 65
        elif state=='ABOVE_A':
            stage='A_PULLBACK' if close[i]<a_hi*.99 else 'A_BREAK'
            score=49 if stage=='A_PULLBACK' else 68
        elif state=='FAILED':stage='FAILED';score=8
        if np.isfinite(depth) and .382<=depth<=.618:score+=8
        if np.isfinite(vr[i]) and vr[i]>=1.5:score+=7
        if np.isfinite(rsi[i]) and 50<=rsi[i]<=70:score+=6
        if np.isfinite(mh[i]) and mh[i]>0:score+=5
        if i>=5 and np.isfinite(ma20[i]) and ma20[i]>ma20[i-5]:score+=4
        if np.isfinite(b_res) and stage=='NEAR_B':score+=5
    row=dict(symbol=code,date=last.date.date().isoformat(),close=_f(last.close),
        stage=stage,stage_zh=CATEGORIES[stage],priority_score=score,
        crash_peak=_f(ph),bottom=_f(p0),a_high=_f(a_hi),b_low=_f(b_lo),
        b_resistance=_f(b_res),distance_to_b_break_pct=near_gap,
        b_retrace=(a_hi-b_lo)/(a_hi-p0) if np.isfinite(a_hi) and np.isfinite(b_lo) and a_hi>p0 else np.nan,
        crash_pct=100*(p0/ph-1) if np.isfinite(ph) else np.nan,
        vol_ratio=_f(last.vol_ratio),rsi=_f(last.rsi),macd_hist=_f(last.macd_hist),
        macd_hist_rising=bool(last.hist_rising),
        ma20_slope5_pct=100*(ma20[-1]/ma20[-6]-1) if np.isfinite(ma20[-1]) and np.isfinite(ma20[-6]) else np.nan,
        ma20=_f(last.ma20),ma60=_f(last.ma60),avg_turnover20=prior20,
        b_break_date=d.iloc[signal_i].date.date().isoformat() if signal_i is not None else '',
        a_break_date=d.iloc[a_break_i].date.date().isoformat() if a_break_i is not None else '',
        bottom_date=d.iloc[bottom_i].date.date().isoformat() if bottom_i is not None else '',
        in_target_summer=recent,qualified=(recent and stage in {'NEAR_B','B_BREAK','C_RUN','A_BREAK','A_PULLBACK'}),
        tradable=bool(np.isfinite(prior20) and prior20>=turnover_limit))
    return row,events,d


def zigzag_five_wave_proxy(d, start, end, entry, bottom):
    """事後 outcome：以兩根右側K確認局部波峰／波谷，並用嚴格的價格非重疊規則過濾。
    僅為五波形態代理指標，非波浪理論公認的唯一數學定義。
    """
    s=d.iloc[start:end+1].reset_index(drop=True)
    if len(s)<14:return False
    highs=s.high.to_numpy(); lows=s.low.to_numpy()
    pivots=[]
    for i in range(2,len(s)-2):
        if highs[i]>max(highs[i-2],highs[i-1],highs[i+1],highs[i+2]):pivots.append((i,'H',highs[i]))
        if lows[i]<min(lows[i-2],lows[i-1],lows[i+1],lows[i+2]):pivots.append((i,'L',lows[i]))
    pivots.sort()
    # 以價格擺動>2.5%排除日內雜訊；同類樞紐只保留更極端者
    clean=[]
    for p in pivots:
        if not clean:
            clean.append(p);continue
        prev=clean[-1]
        if p[1]==prev[1]:
            if (p[1]=='H' and p[2]>prev[2]) or (p[1]=='L' and p[2]<prev[2]):clean[-1]=p
        elif abs(p[2]/prev[2]-1)>=.025:clean.append(p)
    for j in range(max(0,len(clean)-4)):
        p=clean[j:j+5]
        if [v[1] for v in p]!=['H','L','H','L','H']:continue
        h1,l2,h3,l4,h5=[v[2] for v in p]
        w1=h1-bottom;w3=h3-l2;w5=h5-l4
        if (h1>entry*1.025 and l2>bottom and l2<h1*.98 and
            h3>h1*1.02 and l4>h1 and l4<h3*.98 and
            h5>h3*1.01 and w3>=min(w1,w5)):
            return True
    return False


def study_event(d, event, horizon):
    i=event['signal_index'];start=i+1;end=i+horizon
    if end>=len(d):return None
    if _f(d.iloc[start].volume)<=0:return None
    entry=_f(d.iloc[start].open)
    if not np.isfinite(entry) or entry<=0:return None
    # 初始停損不超過8%，若B低點較近就收緊；跳空依開盤成交。
    stop=max(entry*.92, event['b_low']*.995)
    if stop>=entry:return None
    exit_i=end;exit_px=float(d.iloc[end].close);exit_reason='期滿'
    for j in range(start,end+1):
        bar=d.iloc[j]
        if bar.volume<=0:continue
        if bar.open<=stop:
            exit_i=j;exit_px=float(bar.open);exit_reason='跳空停損';break
        if bar.low<=stop:
            exit_i=j;exit_px=stop;exit_reason='結構停損';break
    if d.iloc[exit_i].volume<=0:return None
    net=(exit_px*(1-COST_SELL))/(entry*(1+COST_BUY))-1
    sub=d.iloc[start:exit_i+1];follow=d.iloc[start:end+1]
    # 以下是標的未來走勢結果，不一定表示策略持有至該里程碑。
    reclaim_a=bool((follow.close>=event['a_high']*1.01).any())
    near_crash_peak=bool((follow.high>=event['crash_peak']*.90).any())
    ma20up=follow.ma20>follow.ma20.shift(5)
    # 已於候選日計算的 ma60 為歷史資料，截至每根K當日，不含未來
    strong=(follow.close>follow.ma60)&ma20up&(follow.close>event['a_high']*1.03)
    recover=bool((strong.rolling(5,min_periods=5).sum()>=5).any()
                 and (follow.low.min()>event['b_low']*.995))
    five=zigzag_five_wave_proxy(d,start,end,entry,event['b_low'])
    return dict(**{k:v for k,v in event.items() if k!='signal_index'},horizon=horizon,
        entry_date=d.iloc[start].date.date().isoformat(),entry=entry,
        exit_date=d.iloc[exit_i].date.date().isoformat(),exit=exit_px,
        bars_held=exit_i-start+1,net_pct=100*net,
        mae_pct=100*(sub.low.min()/entry-1),
        mfe_pct=100*(sub.high.max()/entry-1),
        stop_distance_pct=100*(1-stop/entry),exit_reason=exit_reason,
        closed_above_A_high=reclaim_a,near_prior_peak=near_crash_peak,
        trend_recovery=recover,five_wave_proxy=five)


def aggregate(s):
    r=s.net_pct;pos=r[r>0];neg=r[r<0]
    return dict(n=int(len(s)),win_rate_pct=round(100*(r>0).mean(),2),
                avg_net_pct=round(r.mean(),3),med_net_pct=round(r.median(),3),
                profit_factor=round(pos.sum()/(-neg.sum()),3) if len(neg) else np.nan,
                close_break_a_pct=round(100*s.closed_above_A_high.mean(),2),
                recovery_pct=round(100*s.trend_recovery.mean(),2),
                fivewave_proxy_pct=round(100*s.five_wave_proxy.mean(),2),
                avg_mae_pct=round(s.mae_pct.mean(),3))


def make_reports(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=get_data(args)
    if not len(data):raise RuntimeError('市場行情為空')
    asof=data.date.max() if args.asof.lower()=='auto' else pd.Timestamp(args.asof)
    data=data[data.date<=asof].copy()
    if data.empty:raise RuntimeError(f'資料中沒有 {asof.date()} 以前的日線')
    newest=data.date.max();cutoff=newest-pd.Timedelta(days=7)
    symbols=data.groupby('symbol').date.max()
    valid=symbols[symbols>=cutoff].index
    data=data[data.symbol.isin(valid)]
    n_universe=len(valid)
    print(f'[市場] 資料最新日 {newest.date()}，有效股票數 {n_universe}，資料筆數 {len(data):,}',flush=True)
    if n_universe<args.min_universe:
        raise RuntimeError(f'實際可掃描僅 {n_universe} 檔，低於要求 {args.min_universe}。不輸出假200檔。')
    if args.asof.lower()=='auto' and (pd.Timestamp.now().normalize()-newest).days>=8:
        print('[警告] 行情可能過期，請核對資料來源更新時間！',flush=True)
    summer_from=np.datetime64(pd.Timestamp(args.summer_from).date());summer_to=np.datetime64(pd.Timestamp(args.summer_to).date())
    rows=[];events=[];trades=[]
    for idx,(symbol,g) in enumerate(data.groupby('symbol'),1):
        cur,ev,d=inspect_symbol(symbol,g,summer_from,summer_to,args.min_turnover)
        if cur:rows.append(cur)
        for e in ev:
            if not (summer_from<=np.datetime64(pd.Timestamp(e['bottom_date']).date())<=summer_to):continue
            events.append({k:v for k,v in e.items() if k!='signal_index'})
            for h in HORIZONS:
                tr=study_event(d,e,h)
                if tr:trades.append(tr)
        if idx%300==0:print(f'[進度] 分析 {idx} / {n_universe} 檔',flush=True)
    all_df=pd.DataFrame(rows)
    all_df=all_df.sort_values(['priority_score','avg_turnover20'],ascending=False)
    all_df['rank']=np.arange(1,len(all_df)+1)
    all_df.to_csv(out/'all_scanned_stocks.csv',index=False,encoding='utf-8-sig')
    # 觀察200檔不代表200檔都有交易信號：欄位 qualified 和 stage 必須保留
    all_df.head(max(args.watch_size,200)).to_csv(out/'watchlist_top_200_plus.csv',index=False,encoding='utf-8-sig')
    strict=all_df[(all_df.qualified)&(all_df.tradable)]
    strict[strict.stage.isin(['NEAR_B','B_BREAK'])].to_csv(out/'B_near_or_breakout.csv',index=False,encoding='utf-8-sig')
    strict[strict.stage.isin(['C_RUN','A_BREAK','A_PULLBACK'])].to_csv(out/'C_progress_and_A_break.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(events).to_csv(out/'historical_breakout_events.csv',index=False,encoding='utf-8-sig')
    t=pd.DataFrame(trades)
    t.to_csv(out/'matured_trades.csv',index=False,encoding='utf-8-sig')
    summary=[];subgroups=[]
    if not t.empty:
        for (kind,hold),sub in t.groupby(['signal_type','horizon']):
            summary.append(dict(signal_type=kind,horizon=int(hold),**aggregate(sub)))
        q=t[(t.horizon==40)&(t.signal_type=='B_上緣突破')].copy()
        if len(q):
            q['rsi_range']=pd.cut(q.rsi,[0,45,55,65,75,100],include_lowest=True)
            q['volume_multiple']=pd.cut(q.vol_ratio,[0,1.2,1.5,2,3,100],include_lowest=True)
            q['B_retrace_band']=pd.cut(q.b_retrace,[0,.382,.50,.618,.786,1],include_lowest=True)
            q['MACD_up']=q.macd_rising.astype(str)
            q['B_volume_change']=pd.cut(q.b_vol_contraction,[0,.65,.85,1,1.3,10],include_lowest=True)
            for metric in ['rsi_range','volume_multiple','B_retrace_band','MACD_up','B_volume_change']:
                for bucket,s in q.groupby(metric,observed=True):
                    if len(s)<5:continue
                    subgroups.append(dict(metric=metric,band=str(bucket),**aggregate(s),
                                     robust_n30=(len(s)>=30)))
    pd.DataFrame(summary).to_csv(out/'strategy_performance.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(subgroups).to_csv(out/'feature_40day_comparison.csv',index=False,encoding='utf-8-sig')
    nB=int(len(strict[strict.stage.isin(['NEAR_B','B_BREAK'])]));nC=int(len(strict[strict.stage.isin(['C_RUN','A_BREAK','A_PULLBACK'])]))
    with (out/'RESULT_SUMMARY.md').open('w',encoding='utf-8') as f:
        f.write(f'# 台股 B突破→C→回升趨勢量化掃描\n\n'
                f'- 指定截至：{asof.date()}，實際行情最後日：{newest.date()}\n'
                f'- 成功掃描且行情近期的股票：**{n_universe}** 檔\n'
                f'- 股票排名檔：**{len(all_df.head(max(200,args.watch_size)))}** 檔（包括未符合型態者）\n'
                f'- 近B突破／剛突破合格：**{nB}** 檔\n'
                f'- C進行中／站上A高合格：**{nC}** 檔\n'
                f'- 歷史符合訊號：**{len(events)}** 筆；成熟交易紀錄（各持有期合計）：**{len(t)}** 筆\n\n'
                '- 已排除最新行情落後市場7日以上的個股；若來源有倖存者偏差仍需另行校正。\n'
                '- B上緣：不含今日的前12根K高點；B回撤0.25~0.94的反彈區。\n'
                '- B進場：收盤突破B上緣0.5%，量比至少1.2、RSI 46~76、MACD柱體上升，收盤站上MA20。\n'
                '- 真正的成交：訊號下一交易日開盤；結構停損/B低點與最大8%停損取較近者。\n'
                '- 持有10/20/40/60交易日；買賣費用、證交稅與滑價合計納入。\n'
                '- 回升趨勢事件：期間出現5根連續收盤在MA60上且MA20向上，收盤超過A高3%，並且未下破B低點。\n'
                '- 五波代理：事後以2根右側K確認波峰谷與嚴格不重疊檢查，僅是可重複運算的形態代理，不等於Elliott主觀認定。\n'
                '- 事件成功率和買進後的實際獲利率不是同一件事，且重疊股票／事件不是獨立樣本。\n'
                '- 使用最新市場存活股票會有倖存者偏誤；上市櫃、除權息調整和交易限價需另驗證。\n')
    print(f'\n[完成] 實際掃描 {n_universe}，排名 {min(len(all_df),max(200,args.watch_size))}，B候選 {nB}，C候選 {nC}，歷史事件 {len(events)}，成熟交易 {len(t)}')
    if summary:print(pd.DataFrame(summary).to_string(index=False))
    print(f'[產出] {out.resolve()}')
    return {'scanned':n_universe,'ranked':len(all_df.head(max(200,args.watch_size))),
            'B':nB,'C':nC,'signals':len(events),'matured':len(t)}


def main():
    p=argparse.ArgumentParser(description='台股200+股票 B突破→C波→回升趨勢量化掃描')
    p.add_argument('--data',default=None,help='已有的合併OHLCV csv/.gz，若省略則下載TWSTOCK GitHub行情')
    p.add_argument('--asof',default='auto',help='YYYY-MM-DD 或 auto（來源最新日）')
    p.add_argument('--out',default='results')
    p.add_argument('--max-symbols',type=int,default=0,help='測試階段限制下載檔數；正式掃描請勿設定')
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--min-universe',type=int,default=200)
    p.add_argument('--watch-size',type=int,default=200)
    p.add_argument('--min-turnover',type=float,default=1.0e7,help='歷史20日均成交金額下限，預設1000萬元')
    p.add_argument('--summer-from',default='2026-06-01')
    p.add_argument('--summer-to',default='2026-08-31')
    a=p.parse_args()
    make_reports(a)

if __name__=='__main__':main()
