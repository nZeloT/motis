package org.example;

import org.example.data.AbstractConnection;
import org.example.data.Stop;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;



public class ResponseDeconstruct {

  class RaptorStop extends Stop {
    public RaptorStop(int idx, JSONObject stop) {
      super(idx, stop);
      if(USE_RAPTOR_IDS)
        this.raptor_stop_id = eva_to_id.get(this.eva_no);
      else
        this.raptor_stop_id = 0L;
    }

    public final long raptor_stop_id;
  }

  class RaptorLeg {
    RaptorStop from;
    RaptorStop to;
  }

  class Trip extends RaptorLeg {
    public Trip(JSONObject trip, List<RaptorStop> conn_stops) {
      var range = (JSONObject) trip.get("range");
      var dbg = (String) trip.get("debug");
      dbg = dbg.substring(0, dbg.indexOf(':'));
      this.trip_dbg = dbg;

      if(USE_RAPTOR_IDS) {
        this.raptor_ids = dbg_to_route_trips.get(dbg);
        if (this.raptor_ids == null) throw new IllegalStateException("Trip!");
      }

      long fromIdx = (Long) range.get("from");
      long toIdx = (Long) range.get("to");

      this.from = conn_stops.get((int) fromIdx);
      this.to = conn_stops.get((int) toIdx);
    }

    String trip_dbg;
    RouteTrips raptor_ids;
  }

  class Footpath extends RaptorLeg {
    public Footpath(RaptorStop from, RaptorStop to) {
      this.from = from;
      this.to = to;
    }
  }

  class RaptorConnection extends AbstractConnection<RaptorStop> {
    @Override
    public RaptorStop newStop(int idx, JSONObject st) {
      return new RaptorStop(idx, st);
    }

    public RaptorConnection(JSONObject conn) {
      super(conn);
      var trips = (JSONArray) conn.get("trips");

      this.trips = new ArrayList<>();
      Trip prev = null;
      for (var t : trips) {
        var tr = (JSONObject) t;
        var curr = new Trip(tr, super.stops);
        if (prev != null) {
          if (!prev.to.eva_no.equals(curr.from.eva_no)) {
            //insert FP
            this.trips.add(new Footpath(prev.to, curr.from));
          }
        }
        prev = curr;
        this.trips.add(curr);
      }
    }

    @Override
    public String toString() {
      var bld  = new StringBuilder()
        .append("Connection with TR: ").append(tripCount).append(";\tMax Occupancy: ").append(this.moc).append(";\tDuration: ").append(durationStr).append("\n")
        .append("==========================================================================================================\n");

      for (int i = 0, tripsSize = trips.size(); i < tripsSize; i++) {
        var leg = trips.get(i);
        // --1: From:  raptor_id (eva No) Name; Arrived here at: motis time (unix time)
        // ---  Using: raptor_route_id Trips: ... (DBG)
        // ---  To:     raptor_id (eva_no) Name; Arriving at: motis time (unix time)
        bld.append(String.format("%3d", i)).append(": ");

        bld.append("From: ").append(String.format("%-60s", String.format("%6d", leg.from.raptor_stop_id) + " (" + leg.from.eva_no + ") " + leg.from.name + ";"));
        bld.append("\tDeparture at: ").append(String.format("%7d", leg.from.motis_dep_time)).append(" (").append(leg.from.unix_dep_time).append(")\n");

        bld.append("     Using: ");//indent
        if (leg instanceof Trip trip) {
          bld.append(String.format("%-28s", trip.trip_dbg));
          if(USE_RAPTOR_IDS) {
            bld.append(" Route: ").append(String.format("%3d", trip.raptor_ids.route_id)).append("; Trip Ids: ");
            var tripIds = trip.raptor_ids.trip_ids;
            bld.append(tripIds.get(0));
            for (int j = 1; j < tripIds.size(); j++) {
              bld.append(", ");
              bld.append(tripIds.get(j));
            }
          }
          bld.append(";\t").append("Inb. Occupancies (").append(trip.to.idx - trip.from.idx).append(")): ");
          var stop = stops.get(trip.from.idx);
          bld.append("[").append(stop.inbound_occ).append("]");
          for(int j = trip.from.idx+1; j <= trip.to.idx; j++) {
            stop = stops.get(j);
            bld.append(", ").append(stop.inbound_occ);
          }
          bld.append("\n");
        } else {
          bld.append("Footpath\n");
        }

        bld.append("     ");//indent
        bld.append("To:   ").append(String.format("%-60s", String.format("%6d", leg.to.raptor_stop_id) + " (" + leg.to.eva_no + ") " + leg.to.name + ";"));
        bld.append("\tArrival at:   ").append(String.format("%7d", leg.to.motis_arr_time)).append(" (").append(leg.to.unix_arr_time).append(")\n\n");

      }

      return bld.toString();
    }

    final List<RaptorLeg> trips;
  }

  static class RouteTrips {
    RouteTrips(long rid, long tid) {
      route_id = rid;
      trip_ids = new ArrayList<>();
      trip_ids.add(tid);
    }

    final long route_id;
    List<Long> trip_ids;
  }

  HashMap<String, Long> eva_to_id;
  HashMap<String, RouteTrips> dbg_to_route_trips;

  ResponseDeconstruct(List<String> eva_to_stops, List<String> dbg_to_routes) {
    if(USE_RAPTOR_IDS) {
      init_eva_to_stops(eva_to_stops);
      init_dbg_to_routes(dbg_to_routes);
    }
  }

  void init_eva_to_stops(List<String> eva_to_stops) {
    this.eva_to_id = new HashMap<>();
    for (var line : eva_to_stops) {
      var split = line.split(";");
      var eva = split[1].substring(split[1].indexOf(':') + 2);
      var sid = split[0].substring(split[0].lastIndexOf(' ') + 1);
      this.eva_to_id.put(eva, Long.parseLong(sid));
    }
  }

  void init_dbg_to_routes(List<String> dbg_to_routes) {
    this.dbg_to_route_trips = new HashMap<>();
    for (var line : dbg_to_routes) {
      var split = line.split(";");
      var full_dbg = split[2].substring(split[2].indexOf(':') + 2);
      var dbg = full_dbg.substring(0, full_dbg.indexOf(':'));
      var route = split[0].substring(split[0].lastIndexOf(' ') + 1);
      var trip = split[1].substring(split[1].lastIndexOf(' ') + 1);

      var entry = this.dbg_to_route_trips.get(dbg);
      if (entry != null) {
        if (Long.parseLong(route) != entry.route_id) throw new IllegalStateException();
        entry.trip_ids.add(Long.parseLong(trip));
        this.dbg_to_route_trips.put(dbg, entry);
      } else {
        this.dbg_to_route_trips.put(dbg, new RouteTrips(Long.parseLong(route), Long.parseLong(trip)));
      }
    }
  }

  List<RaptorConnection> deconstruct(JSONObject response) {
    var content = (JSONObject) response.get("content");
    var conns = (JSONArray) content.get("connections");
    var r = new ArrayList<RaptorConnection>();
    for (var c : conns) {
      var conn = (JSONObject) c;
      var co = new RaptorConnection(conn);
      r.add(co);
    }

    return r;
  }


  public static void main(String[] args) throws IOException, ParseException {
    var responses = Files.readAllLines(Path.of("./responses.txt"));
    var eva_to_stop = new ArrayList<String>();
    var dbg_to_route_trips = new ArrayList<String>();
    if(USE_RAPTOR_IDS) {
      eva_to_stop.addAll(Files.readAllLines(Path.of("./eva_to_stop.txt")));
      dbg_to_route_trips.addAll(Files.readAllLines(Path.of("./dbg_to_route_trips.txt")));
    }
    var deconstructor = new ResponseDeconstruct(eva_to_stop, dbg_to_route_trips);
    var parser = new JSONParser();

    for (var line : responses) {
      var response = (JSONObject) parser.parse(line);
      var id = response.get("id");
      System.out.println("Responses for Query ID " + id);
      System.out.println("==========================================================================================================");
      var conns = deconstructor.deconstruct(response);
      for (var c : conns) {
        System.out.println(c);
        System.out.println();
        System.out.println();
      }
      System.out.println();
      System.out.println();
    }

  }

  static final boolean USE_RAPTOR_IDS = false;
}
