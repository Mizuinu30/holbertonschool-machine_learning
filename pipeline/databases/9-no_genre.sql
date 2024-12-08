-- Lists all TV shows without a genre linked
-- Results are sorted by tv_shows.title and tv_show_genres.genre_id

SELECT
    tv_shows.title AS title,
    NULL AS genre_id
FROM
    tv_shows
LEFT JOIN
    tv_show_genres
ON
    tv_shows.id = tv_show_genres.show_id
WHERE
    tv_show_genres.genre_id IS NULL
ORDER BY
    tv_shows.title ASC,
    genre_id ASC;
