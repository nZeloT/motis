package org.example;

public class Utils {

  static final long SCHEDULE_BEGIN = 1629504000;
  public static long unix_to_motis_time(long unix_time) {
    if (unix_time < SCHEDULE_BEGIN) {
      throw new IllegalStateException("Time!");
    }
    return (unix_time - SCHEDULE_BEGIN) / 60;
  }
}
