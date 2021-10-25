package org.example;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class EmptyResultLister {

  public static void main(String[] args) throws IOException, ParseException {

    var resLines = Files.readAllLines(Path.of("./data/results/r-fwd-raptor_cpu-mod-moc.txt"));

    var parser = new JSONParser();

    for(var line : resLines) {
      var response = (JSONObject)parser.parse(line);
      var content = (JSONObject)response.get("content");
      var conns = (JSONArray)content.get("connections");
      if(conns.isEmpty()) {
        var id = (Long)response.get("id");
        System.out.println("Found empty response with ID: " + id);
      }
    }

  }

}
