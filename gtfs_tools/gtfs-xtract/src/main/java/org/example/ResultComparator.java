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
import java.util.*;

public class ResultComparator {

  static class CompareConnection extends AbstractConnection<Stop> {
    public final long query_id;
    public List<Integer> dominatesConnIdx;

    @Override
    public Stop newStop(JSONObject st) {
      return new Stop(st);
    }

    public CompareConnection(long id, JSONObject connection) {
      super(connection);
      this.query_id = id;
    }

    public String toString(boolean mocRelevant) {
      return departureFmt() + "\t\t" + arrivalFmt() + "\t\t" + durationStr + "\tTR: " + tripCount + (mocRelevant ? "\tMOC: " + moc : "");
    }

    public boolean dominates(CompareConnection toDominate, boolean mocRelevant) {
      if(toDominate == null) return true;

      return tripCount <= toDominate.tripCount && (!mocRelevant || moc <= toDominate.moc) && unix_arr_time <= toDominate.unix_arr_time;
    }

    public int compareTo(CompareConnection o, boolean mocRelevant) {
      if (o == null) return -1;

      var tc = Long.compare(this.tripCount, o.tripCount);
      if (tc != 0) return tc;

      var mc = Long.compare(this.moc, o.moc);
      if (mc != 0 && mocRelevant) return mc;

      var dc = Long.compare(this.duration.getSeconds(), o.duration.getSeconds());
      if (dc != 0) return dc;

      return Long.compare(this.unix_arr_time, o.unix_arr_time);
    }
  }

  static class ComparisonResult {

    public final long id;

    List<CompareConnection> raptorConns;
    List<CompareConnection> routingConns;

    int matchingConns;
    int rpcConCnt = 0;
    int rocConCnt = 0;

    int rpcTrMocMask = 0;
    int rocTrMocMask = 0;

    public ComparisonResult(long id) {
      this.raptorConns = new ArrayList<>();
      this.routingConns = new ArrayList<>();
      this.id = id;
      this.matchingConns = 0;
    }

    void addMatch(CompareConnection rpc, CompareConnection roc) {
      raptorConns.add(rpc);
      routingConns.add(roc);
      ++rocConCnt;
      ++rpcConCnt;
      ++matchingConns;
      rpcTrMocMask = _update_mask(rpcTrMocMask, rpc);
      rocTrMocMask = _update_mask(rocTrMocMask, roc);
    }

    void addRpcOnly(CompareConnection rpc) {
      raptorConns.add(rpc);
      routingConns.add(null);
      ++rpcConCnt;
      rpcTrMocMask = _update_mask(rpcTrMocMask, rpc);
    }

    void addRocOnly(CompareConnection roc) {
      raptorConns.add(null);
      routingConns.add(roc);
      ++rocConCnt;
      rocTrMocMask = _update_mask(rocTrMocMask, roc);
    }

    int _update_mask(int mask, CompareConnection c) {
      final var tr = c.tripCount;
      final var moc = c.moc;

      final var combined = (1 << (tr * (moc+1)));
      mask = mask | combined;
      return mask;
    }

    boolean isFullMatch() {
      return matchingConns == Math.max(raptorConns.size(), routingConns.size());
    }

    boolean isFullMatchOnTripsAndMoc() {
      return rpcTrMocMask == rocTrMocMask;
    }

    public String toString(boolean mocRelevant) {
      if (routingConns.size() != raptorConns.size()) throw new IllegalStateException("Mismatch conn count!");
      var empty = String.format("%-74s", "---");
      var bld = new StringBuilder();
      bld.append("Comparison for Query ID: ").append(id).append(";\tFound full match: ").append(isFullMatch()).append(";\tMatching Connection Count: ").append((rpcConCnt == rocConCnt)).append("\n");
      bld.append("==================================================================================================\n");
      for (int i = 0; i < raptorConns.size(); i++) {
        var lhs = raptorConns.get(i);
        var rhs = routingConns.get(i);

        var lhsString = lhs != null ? lhs.toString(mocRelevant) : empty;
        var rhsString = rhs != null ? rhs.toString(mocRelevant) : empty;
        var matches = (lhs != null && rhs != null && lhs.compareTo(rhs, mocRelevant) == 0) ? "M" : "-";

        bld.append(String.format("%02d", i)).append(": ").append(lhsString).append("\t\t").append(matches).append("\t\t").append(rhsString).append("\n");
      }

      return bld.toString();
    }
  }

  static List<CompareConnection> getConnections(long id, JSONObject content, boolean mocRelevant) {
    var conns = (JSONArray) content.get("connections");
    var cs = new ArrayList<CompareConnection>();
    for (var c : conns) {
      var connection = (JSONObject) c;
      cs.add(new CompareConnection(id, connection));
    }
    cs.sort((lhs, rhs) -> {
      if (lhs == null || rhs == null) throw new IllegalStateException("Received a null!");
      return lhs.compareTo(rhs, mocRelevant);
    });
    return cs;
  }

  static HashMap<Long, List<CompareConnection>> transform(List<String> lines, boolean mocRelevant) throws ParseException {
    var parser = new JSONParser();
    var map = new HashMap<Long, List<CompareConnection>>();
    for (var line : lines) {
      var response = (JSONObject) parser.parse(line);
      var id = (Long) response.get("id");
      var conns = getConnections(id, (JSONObject) response.get("content"), mocRelevant);
      map.put(id, conns);
    }
    return map;
  }

  static List<ComparisonResult> compare(long resCount, Map<Long, List<CompareConnection>> raptorConns, Map<Long, List<CompareConnection>> routingConns, boolean mocRelevant) {

    var r = new ArrayList<ComparisonResult>();
    for (long queryId = 1; queryId <= resCount; queryId++) {

      var rpc = raptorConns.get(queryId);
      var roc = routingConns.get(queryId);

      var result = new ComparisonResult(queryId);

      var rpcIdx = 0;
      var rocIdx = 0;

      while (rpcIdx < rpc.size() && rocIdx < roc.size()) {

        var rapc = rpc.get(rpcIdx);
        var rouc = roc.get(rocIdx);

        var compare = rapc.compareTo(rouc, mocRelevant);
        if (compare < 0) {
          // LHS better than RHS
          result.addRpcOnly(rapc);
          ++rpcIdx;
        } else if (compare > 0) {
          // RHS better than LHS
          result.addRocOnly(rouc);
          ++rocIdx;
        } else {
          // match
          result.addMatch(rapc, rouc);
          ++rpcIdx;
          ++rocIdx;
        }
      }

      while (rpcIdx < rpc.size()) {
        result.addRpcOnly(rpc.get(rpcIdx));
        ++rpcIdx;
      }

      while (rocIdx < roc.size()) {
        result.addRocOnly(roc.get(rocIdx));
        ++rocIdx;
      }

      r.add(result);
    }

    return r;
  }

  public static void main(String[] args) throws IOException, ParseException {
    var mocRelevant = true;
    var raptorLines = Files.readAllLines(Path.of("./data/results/r-fwd-raptor_cpu-moc.txt"));
    var routingLines = Files.readAllLines(Path.of("./data/results/r-fwd-routing-moc.txt"));
    //var routingLines = Files.readAllLines(Path.of("./data/results/r-fwd-raptor_cpu-mod-moc_bi.txt"));

    if (raptorLines.size() != routingLines.size()) throw new IllegalStateException("Line Counts don't match!");

    var raptorConns = transform(raptorLines, mocRelevant);
    var routingConns = transform(routingLines, mocRelevant);

    var comparison = compare(raptorLines.size(), raptorConns, routingConns, mocRelevant);

    var full_match_count = 0;
    var matchingConCount = 0;
    var totalMatchingCnt = 0;
    var totalConnCnt = 0;
    var matchingTrMoc = 0;
    var moreRpcConns = new ArrayList<ComparisonResult>();
    var moreRocConns = new ArrayList<ComparisonResult>();
    var totalCount = raptorLines.size();

    for (var res : comparison) {
      System.out.println(res.toString(mocRelevant));
      System.out.println();
      System.out.println();

      totalMatchingCnt += res.matchingConns;
      totalConnCnt += res.raptorConns.size();

      if (res.isFullMatch())
        ++full_match_count;

      if(res.isFullMatchOnTripsAndMoc())
        ++matchingTrMoc;

      if (res.rocConCnt == res.rpcConCnt)
        ++matchingConCount;

      if(res.rpcConCnt > res.rocConCnt)
        moreRpcConns.add(res);

      if(res.rocConCnt > res.rpcConCnt)
        moreRocConns.add(res);
    }

    System.out.println("Statistics:");
    System.out.println(String.format("%35s", "Full Matches: ") + "\t" + String.format("%4d", full_match_count) + "/" + totalCount + ";\t" + String.format("%.2f", (full_match_count+0.0)/totalCount));
    System.out.println(String.format("%35s", "ConnCnt Matches (TR,MOC): ") + "\t" + String.format("%4d", matchingTrMoc) + "/" + totalCount + ";\t" + String.format("%.2f", (matchingTrMoc+0.0)/totalCount));
    System.out.println(String.format("%35s", "Total Conn. Matches: ") + "\t" + String.format("%4d", totalMatchingCnt) + "/" + totalConnCnt + ";\t" + String.format("%.2f", (totalMatchingCnt+0.0)/totalConnCnt));
    System.out.println();
    System.out.println(String.format("%35s", "Connection Count Matches: ") + "\t" + String.format("%4d", matchingConCount) + "/" + totalCount + ";\t" + String.format("%.2f", (matchingConCount+0.0)/totalCount));
    System.out.println();;
    System.out.println(String.format("%35s", "More Raptor Conns: ") + "\t" + String.format("%4d", moreRpcConns.size()) + "/" + totalCount);
    System.out.println(String.format("%35s", "More Routing Conns: ") + "\t" + String.format("%4d", moreRocConns.size()) + "/" + totalCount);
    System.out.println();
    System.out.println();

    var blcRpc = new StringBuilder("More Raptor Connections for Queries: ");
    moreRpcConns.forEach((e) -> blcRpc.append(e.id).append(", "));
    System.out.println(blcRpc);
    //System.out.println();
    //moreRpcConns.forEach((e) -> System.out.println(e.toString(mocRelevant) + "\n\n"));
    System.out.println();
    System.out.println();

    var blcRoc = new StringBuilder("More Routing Connections for Queries: ");
    moreRocConns.forEach((e) -> blcRoc.append(e.id).append(", "));
    System.out.println(blcRoc);
  }

}
