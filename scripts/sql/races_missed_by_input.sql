.headers on
.mode csv
.output races_missed_by_input.csv


SELECT DISTINCT 
  REPLACE(h2.code, '_hirace', '') as code,
  h2.graph
FROM
  hirace h1
  inner join hirace h2
    on h1.code = h2.code
    and h1.graph <> h2.graph
WHERE
  h1.errors > 0
  and h2.errors = 0
  and h2.graph LIKE '%5n_5e%'
;
