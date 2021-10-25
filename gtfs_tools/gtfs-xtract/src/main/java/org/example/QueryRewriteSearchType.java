package org.example;

import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public class QueryRewriteSearchType {


  public static void main(String[] args) throws IOException, ParseException {

    var requests = Files.readAllLines(Path.of("./data/queries/queries-fwd-routing.txt"));
    var outBuffer = new ArrayList<String>();

    for(var line : requests) {
      var out = line.replace("Default", "MaxOccupancy");
      outBuffer.add(out);
    }

    Files.write(Path.of("./data/queries/queries-fwd-routing-moc.txt"), outBuffer);
  }
}
