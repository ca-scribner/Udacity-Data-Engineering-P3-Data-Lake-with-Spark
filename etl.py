import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get("AWS", 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get("AWS", 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Create and return a spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, song_data, output_location):
    """
    Process song data using an existing spark session
    
    Results are stored to as songs.parquet and artists.parquet data in output_location
    
    Args:
        spark: A spark session
        song_data (str): Path to song_data file, such as an s3a:// link
        output_location (str): Path to a parent directory for storing processed song data,
                               such as an s3a:// link
    
    Returns:
        None
    """
    print(f"Processing song_data from {song_data}")
    
    # Define song file schema
    song_schema = StructType([
        StructField("song_id", StringType()),
        StructField("title", StringType()),
        StructField("artist_id", StringType()),
        StructField("year", IntegerType()),
        StructField("duration", DoubleType()),
    ])
    
    # read song data file
    print("Reading song data")
    df = spark.read.json(song_data, schema=song_schema)

    # extract columns to create songs table
    songs_table = df.select(
        "song_id", 
        "title", 
        "artist_id", 
        "year", 
        "duration"
    )
    
    # write songs table to parquet files partitioned by year and artist
    songs_output_file = output_location + '/songs.parquet'
    print(f"Writing song data to {songs_output_file}")
    songs_table.write.parquet(songs_output_file, partitionBy=['year', 'artist_id'])

    # extract columns to create artists table
    artists_table = df.select(
        "artist_id", 
        "song_id", 
        "title", 
        "year", 
        "duration"
    )
    
    # write artists table to parquet files
    artists_output_file = output_location + '/artists.parquet'
    print(f"Writing artists to {artists_output_file}")
    artists_table.write.parquet(artists_output_file)


def process_log_data(spark, log_data, output_location):
    """
    Process log data and existing song data using an existing spark session
    
    Results are stored to as users.parquet, time.parquet, and songplays.parquet 
    in output_location
    
    Args:
        spark: A spark session
        log_data (str): Path to log_data file, such as an s3a:// link
        output_location (str): Path to a parent directory for storing processed song data,
                               such as an s3a:// link
    
    Returns:
        None
    """
    print(f"Processing log_data from {log_data}")
    
    # read log data file
    print("Reading log data")
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(F.col("page") == "NextSong")

    # extract columns for users table    
    users_table = df.selectExpr(
        "userId as user_id",
        "firstName as first_name",
        "lastName as last_name",
        "gender",
        "level"
    )
    users_table = users_table.drop_duplicates(["user_id"])
    
    # write users table to parquet files
    users_output_file = output_location + '/users.parquet'
    print(f"Writing users data to {users_output_file}")
    users_table.write.parquet(users_output_file)

    # create datetime column from original timestamp column
    df_datetime = df.select('ts').withColumn('datetime', from_unixtime(F.col("ts") / 1000).cast(TimestampType()))
    # extract columns to create time table
    time_table = df_datetime.select(
        F.col("ts").alias('start_time'),
        hour("datetime").alias('hour'),
        dayofmonth("datetime").alias('day'),
        weekofyear("datetime").alias("week"),
        month("datetime").alias("month"),
        year("datetime").alias("year"),
        date_format("datetime", "u").alias("weekday")
    )
    time_table = time_table.drop_duplicates(["start_time"])
    
    # write time table to parquet files partitioned by year and month
    time_output_file = output_location + '/time.parquet'
    print(f"Writing time data to {users_output_file}")
    time_table.write.parquet(time_output_file, partitionBy=['year', 'month'])

    # read in song data to use for songplays table
    song_data = output_location + '/songs.parquet'
    print(f"Loading processed song_data from {users_output_file}")
    song_df = spark.read.parquet(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    song_columns = ["song_id", "title", "artist_id"]
    time_columns = ["start_time", "year", "month"]
    songplays_table = (df
                       .join(song_df.select(*song_columns), song_df.title == df.song)
                       .join(time_table.select(*time_columns), time_table.start_time == df.ts)
                       .select(
                           F.col("ts").alias("start_time"),
                           F.col("userId").alias("user_id"),
                           "level",
                           "song_id",
                           "artist_id",
                           F.col("sessionId").alias("session_id"),
                           "location",
                           F.col("userAgent").alias("user_agent"),
                           "year",
                           "month",
                       )
                      )

    # write songplays table to parquet files partitioned by year and month
    songplays_output_file = output_location + '/songplays.parquet'
    print(f"Writing songplay data to {songplays_output_file}")
    songplays_table.write.parquet(songplays_output_file, partitionBy=["year", "month"])


def main():
    spark = create_spark_session()
    
    lake_config = configparser.ConfigParser()
    lake_config.read('lake.cfg')
    
    output_location = lake_config.get("S3", "PROCESSED_DATA_LOCATION")
    input_song_data = lake_config.get("S3", "SONG_DATA")
    process_song_data(spark, input_song_data, output_location)   
    
    input_log_data = lake_config.get("S3", "LOG_DATA")
    process_log_data(spark, input_log_data, output_location)


if __name__ == "__main__":
    main()
