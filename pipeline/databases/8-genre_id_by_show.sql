-- Lists all shows in hbtn_0d_tvshows that have at least one genre linked
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT
    tv_shows.title AS title,
    tv_show_genres.genre_id AS genre_id
FROM
    tv_shows
JOIN
    tv_show_genres
ON
    tv_shows.id = tv_show_genres.show_id
ORDER BY
tv_shows.title ASC,
tv_show_genres.genre_id ASC;
