sqlContext.sql("""CREATE TEMPORARY TABLE tmp_tracks_by_album
        USING org.apache.spark.sql.cassandra
        OPTIONS (
          keyspace "music",
          table "tracks_by_album",
          pushdown "true"
        )""")

track_count_by_year = sqlContext.sql("select 'dummy' as dummy, album_year as year, count(*) as track_count from tmp_tracks_by_album group by album_year")

track_count_by_year.show(5)

track_count_by_year.registerTempTable("tmp_track_count_by_year")

sqlContext.sql("insert into table music.tracks_by_year select dummy, track_count, year from tmp_track_count_by_year")

sqlContext.sql("insert into table music.tracks_by_year select 'dummy' as dummy, count(*) as track_count, album_year from tmp_tracks_by_album group by album_year")



