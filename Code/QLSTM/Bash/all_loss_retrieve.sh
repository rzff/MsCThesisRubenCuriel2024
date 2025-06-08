echo "jobname,jobid,fold,variant,epoch,batch,total_batches,loss,filename" > all_losses.csv; \
for f in *.err; do \
  base="${f%.err}"; \
  jn="${base%%_*}"; \
  jid="${base#*_}"; jid="${jid%%_*}"; \
  fold="${base##*_}"; \
  tr '\r' '\n' <"$f" | awk -v jn="$jn" -v jid="$jid" -v fold="$fold" -v fn="$f" '\
    BEGIN { OFS=","; prev_ep=-1; switched=0 }\
    {\
      # 1) detect any Epoch N and update switched when resets\
      if (match($0,/Epoch[[:space:]]*([0-9]+)/,e)) {\
        ep=e[1]+0;\
        if (prev_ep>=0 && ep<prev_ep) switched=1;\
        prev_ep=ep;\
      }\
      \
      # 2a) classical metrics: “RMSE:” lines → treat RMSE as loss\
      if (/RMSE:/ && match($0,/RMSE:[[:space:]]*([0-9]+\.[0-9]+)/,r)) {\
        print jn, jid, fold, (switched?"quantum":"classical"), prev_ep, "", "", r[1], fn;\
      }\
      \
      # 2b) quantum per-batch losses: “loss=” lines\
      if (/loss=/ && match($0,/([0-9]+)\/([0-9]+).*loss=([0-9]+\.[0-9]+)/,m)) {\
        print jn, jid, fold, (switched?"quantum":"classical"), prev_ep, m[1], m[2], m[3], fn;\
      }\
    }' >> all_losses.csv; \
done && echo "✔ Done — see $(pwd)/all_losses.csv"
