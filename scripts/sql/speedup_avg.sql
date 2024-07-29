.headers on
.mode column

SELECT
    g.graph,
    g.threads_per_block,
    g.number_of_blocks,
    AVG(g.time_ns) / 1000 as iguard_ms_avg,
    AVG(h.time_ns) / 1000 as hirace_ms_avg,
    AVG(g.time_ns) / AVG(h.time_ns) as hirace_avg_speedup
FROM iguard g
    LEFT JOIN (
      SELECT 
        REPLACE(code, '_hirace', '') AS code,
        graph,
        threads_per_block,
        number_of_blocks,
        time_ns
      FROM
        hirace
    ) AS h
        ON  g.code = h.code
        AND g.graph = h.graph
        AND g.threads_per_block = h.threads_per_block
        AND g.number_of_blocks = h.number_of_blocks
GROUP BY
    g.graph,
    g.threads_per_block,
    g.number_of_blocks;
