import test from "node:test";
import assert from "node:assert/strict";
import worker, {aggregate} from "../src/index.js";

test("statistical reducer keeps measured and unmeasured horizons separate", () => {
  const stats = aggregate({schema_version:"trend_breakout_shadow_v1",events:[
    {symbol:"ABCUSDT",direction:"YUKARI",reference_price:100,
     max_up_pct:3,max_down_pct:-2,first_plus_1pct_at:100,first_minus_1pct_at:101,
     coverage_gaps:0,signal_context:{risk_count:1},
     horizons:{"24h":{close_pct:1.2},"48h":null,"72h":null}},
    {symbol:"XYZUSDT",direction:"AŞAĞI",reference_price:10,
     max_up_pct:2,max_down_pct:-4,first_plus_1pct_at:null,first_minus_1pct_at:200,
     coverage_gaps:1,signal_context:{risk_count:3},
     horizons:{"24h":{close_pct:-2},"48h":null,"72h":null}},
  ]});
  assert.equal(stats.sample_count,2);
  assert.equal(stats.by_direction.YUKARI.count,1);
  assert.equal(stats.by_direction.AŞAĞI.horizons["24h"].mean_pct,-2);
  assert.equal(stats.by_direction.AŞAĞI.horizons["72h"].measured,0);
  assert.equal(stats.by_risk_count[1].count,1);
  assert.equal(stats.by_risk_count[3].count,1);
  assert.equal(stats.worst_adverse_moves[0].symbol,"XYZUSDT");
});

test("rejects unrecognized datasets", () => {
  assert.throws(()=>aggregate({schema_version:"wrong",events:[]}),/INVALID_TREND_SCHEMA/);
});

test("daily empty source writes deterministic research report without AI call",async()=>{
  const writes=[];
  const env={
    R2_PREFIX:"crash-bounce",
    ARCHIVE_BUCKET:{
      head:async()=>null,
      get:async(key)=>{
        assert.equal(key,"crash-bounce/trend-breakout/2026/11/11/trend_breakout_2026-11-11.json");
        return {size:100,json:async()=>({schema_version:"trend_breakout_shadow_v1",events:[]})};
      },
      put:async(key,body,options)=>{writes.push({key,data:JSON.parse(body),options});}
    },
    AI:{run:async()=>{throw Error("not called on empty research");}}
  };
  const pending=[];
  const scheduledTime=Date.UTC(2026,10,11,20,45);
  await worker.scheduled({scheduledTime},env,{waitUntil(p){pending.push(p);}});
  await Promise.all(pending);
  assert.equal(writes.length,1);
  assert.equal(writes[0].data.statistics.sample_count,0);
  assert.equal(writes[0].data.research_only,true);
  assert.equal(writes[0].data.trading_execution,false);
  assert.match(writes[0].key,/ai-reports/);
});
