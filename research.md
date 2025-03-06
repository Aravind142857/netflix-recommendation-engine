# Research
## Type of recommendation filtering
    1. Collaborative filtering - Suggests items/content that other similar users have liked. Similarity of users is determined based on purchase history, ratings, etc. Could face cold start problem for new users.
    2. Content-based filtering - Suggests items/content based on features of items/content that the user has interacted with or liked previously. Features of a movie/series include actors, duration, type (movie or series), maturity rating, plot, etc.

## Data
    1. show_id: uuid for each series/movie
    2. title: name of each series/movie
    3. director: Director(s) of the series/movie
    4. cast: List of actors
    5. country: Country where the series/movie was produced
    6. date_added: Date when the series/movie was added to Netflix
    7. release_year: Year the series/movie was released
    8. rating: Maturity rating of series/movie
    9. duration: duration of series/movie (number of seasons for series, minutes for movies)
    10. listed_in: Genres or categories that the series/movie belongs to
    11. description: Description of the series/movie
## Data processing
    Replace null values or drop the row if the column is essential to the recommendation engine
