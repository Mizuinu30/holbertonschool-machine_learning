-- Lists all genres with the number of shows linked
-- Results are sorted by the number of shows linked in descending order

SELECT
    genres.name AS genre,
    COUNT(tv_show_genres.show_id) AS number_of_shows
FROM
    genres
JOIN
    tv_show_genres
ON
    genres.id = tv_show_genres.genre_id
GROUP BY
    genres.id, genres.name
ORDER BY
    number_of_shows DESC;
