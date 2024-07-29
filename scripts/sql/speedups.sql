.headers on
.mode csv
.output speedups.csv

SELECT
  i.code,
  i.graph,
  -- i.threads_per_block,
  -- i.number_of_blocks,
  i.time_ns AS indigo_runtime_n,
  h.time_ns AS hirace_runtime_n,
  g.time_ns AS iguard_runtime_n,
  g.time_ns / i.time_ns AS iguard_overhead,
  h.time_ns / i.time_ns AS hirace_overhead,
  g.time_ns / h.time_ns AS hirace_speedup
FROM
  indigo i
  LEFT JOIN iguard g
    ON i.code = g.code
      AND i.graph = g.graph
      AND i.threads_per_block = g.threads_per_block
      AND i.number_of_blocks = g.number_of_blocks
  LEFT JOIN hirace h
    ON i.code = REPLACE(h.code, "_hirace", "")
      AND i.graph = h.graph
      AND i.threads_per_block = h.threads_per_block
      AND i.number_of_blocks = h.number_of_blocks
GROUP BY
  i.code,
  i.graph,
  i.threads_per_block,
  i.number_of_blocks
;
