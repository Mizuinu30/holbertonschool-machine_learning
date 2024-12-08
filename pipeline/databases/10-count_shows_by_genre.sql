-- Lists all genres with the number of shows linked
-- Results are sorted by the number of shows linked in descending order

SELECT
    tg.name AS genre,
    COUNT(tsg.genre_id) AS number_of_shows
FROM
    tv_genres tg, tv_show_genres tsg
WHERE
    tg.id = tsg.genre_id
GROUP BY
    genre
ORDER BY
    number_of_shows DESC;
