

import click

from twitter import fs
from twitter.utils import get_spark, try_or_none
from twitter.models import GeoTweet

from litecoder import twitter_usa_city_state


@try_or_none
def geocode(tweet):
    """Try to find a city / state.
    """
    city, state = twitter_usa_city_state(tweet.actor.location)

    if not (city or state):
        return

    kwargs = dict(
        id=tweet.id,
        posted_time=tweet.posted_time,
        actor_id=tweet.actor.id,
        location=tweet.actor.location,
        body=tweet.body,
        twitter_lon=tweet.geo.lon,
        twitter_lat=tweet.geo.lat,
    )

    if state:
        kwargs['state'] = state.abbr

    if city:
        kwargs['city_geonameid'] = city.geonameid
        kwargs['city_name'] = city.name
        kwargs['population'] = city.population
        kwargs['geonames_lon'] = city.longitude
        kwargs['geonames_lat'] = city.latitude

    return GeoTweet(**kwargs)


@click.command()
@click.option('--src', default='/data/tweets.parquet')
@click.option('--dest', default='/data/geo-tweets.parquet')
def main(src, dest):
    """Geocode tweets.
    """
    sc, spark = get_spark()

    tweets = spark.read.parquet(src)

    matches = tweets.rdd \
        .filter(lambda t: t.actor.location and t.actor.language == 'en') \
        .map(geocode) \
        .filter(bool) \
        .toDF(GeoTweet.schema)

    matches.write \
        .mode('overwrite') \
        .parquet(dest)


if __name__ == '__main__':
    main()
