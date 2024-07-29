.headers on
.mode column


SELECT
    i.graph,
    i.threads_per_block,
    i.number_of_blocks,
    COUNT(DISTINCT i.code) AS num_tests,
    SUM(CASE WHEN h.code IS NOT NULL THEN 1 ELSE 0 END) AS hirace_result_count,
    SUM(CASE WHEN g.code IS NOT NULL THEN 1 ELSE 0 END) AS iguard_result_count,
    SUM(CASE WHEN m.code IS NOT NULL THEN 1 ELSE 0 END) AS memcheck_result_count,
    SUM(CASE WHEN h.errors > 0 THEN 1 ELSE 0 END) AS hirace_error_found,
    SUM(CASE WHEN g.errors > 0 THEN 1 ELSE 0 END) AS iguard_error_found,
    SUM(CASE WHEN m.errors > 0 THEN 1 ELSE 0 END) AS memcheck_error_found,
    SUM(CASE WHEN i.errors > 0 THEN 1 ELSE 0 END) AS indigo_error_found
FROM indigo i
    LEFT JOIN iguard g
        ON  i.code = g.code
        AND i.graph = g.graph
        AND i.threads_per_block = g.threads_per_block
        AND i.number_of_blocks = g.number_of_blocks
    LEFT JOIN memcheck m
        ON m.code = i.code
        AND m.graph = i.graph
        AND m.threads_per_block = i.threads_per_block
        AND m.number_of_blocks = i.number_of_blocks
        AND m.tool = 'memcheck'
    LEFT JOIN (
      SELECT 
        REPLACE(code, '_hirace', '') AS code,
        graph,
        threads_per_block,
        number_of_blocks,
        errors
      FROM
        hirace
    ) AS h
        ON i.code = h.code
        AND i.graph = h.graph
        AND i.threads_per_block = h.threads_per_block
        AND i.number_of_blocks = h.number_of_blocks
WHERE
  h.code IS NOT NULL
GROUP BY
    i.graph,
    i.threads_per_block,
    i.number_of_blocks;
