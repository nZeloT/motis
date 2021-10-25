package org.example;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class OccupancyStats {

  static class RequestOccupancy {

    RequestOccupancy(long id) {
      this.req_id = id;
      this.max_occupancies = new ArrayList<>();
    }

    @Override
    public String toString() {
      return "id: " + String.format("%5d", req_id) + "; Max Occupancies: "
        + max_occupancies.stream().map(e -> String.format("%d", e)).reduce("", (a, b) -> b + ", " + a);
    }

    long req_id;
    List<Long> max_occupancies;
  }

  static long max_occupancy_of_connection(JSONObject connection) {
    return (Long) connection.get("max_occupancy");
  }

  static RequestOccupancy process_response(JSONObject response) {
    var id = (Long) response.get("id");
    var content = (JSONObject) response.get("content");
    var connections = (JSONArray) content.get("connections");

    var occ = new RequestOccupancy(id);

    for (var c : connections) {
      var conn = (JSONObject) c;
      occ.max_occupancies.add(max_occupancy_of_connection(conn));
    }
    return occ;
  }

  public static void main(String[] args) throws IOException, ParseException {

    var result_lines = Files.readAllLines(Path.of("./data/results/r-fwd-raptor_cpu-mod.txt"));
    var parser = new JSONParser();
    for (var line : result_lines) {
      var response = (JSONObject)parser.parse(line);
      System.out.println(process_response(response));
    }

  }

}
