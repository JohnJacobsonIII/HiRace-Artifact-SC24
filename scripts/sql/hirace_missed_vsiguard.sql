.headers on
.mode column


SELECT
    DISTINCT i.code
FROM indigo i
    LEFT JOIN iguard g
        ON  i.code = g.code
        AND i.graph = g.graph
        AND i.threads_per_block = g.threads_per_block
        AND i.number_of_blocks = g.number_of_blocks
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
  h.errors = 0
  AND g.errors > 0
;
