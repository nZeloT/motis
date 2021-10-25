package org.example.data;

import org.example.Utils;
import org.json.simple.JSONObject;

public class Stop {
  public Stop(JSONObject stop) {
    var station = (JSONObject) stop.get("station");
    var arrival = (JSONObject) stop.get("arrival");
    var departure = (JSONObject) stop.get("departure");

    this.eva_no = (String) station.get("id");
    this.name = (String) station.get("name");

    this.unix_arr_time = (Long) arrival.get("time");
    if(this.unix_arr_time > 0)
      this.motis_arr_time = Utils.unix_to_motis_time(this.unix_arr_time);
    else
      this.motis_arr_time = 0;

    this.unix_dep_time = (Long) departure.get("time");
    if(this.unix_dep_time > 0)
      this.motis_dep_time = Utils.unix_to_motis_time(this.unix_dep_time);
    else
      this.motis_dep_time = 0;
  }

  public final long unix_arr_time;
  public final long motis_arr_time;
  public final long unix_dep_time;
  public final long motis_dep_time;
  public final String eva_no;
  public final String name;
}
