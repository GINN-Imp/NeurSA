import java.util.*;

import fr.inria.controlflow.*;
import org.json.simple.JSONArray;
import spoon.reflect.code.CtCodeElement;
import spoon.reflect.code.CtStatement;
import spoon.reflect.declaration.CtElement;
import org.json.simple.JSONObject;
import spoon.reflect.declaration.CtModifiable;
import spoon.reflect.declaration.CtTypedElement;


public class intervalNode {
    HashSet<ControlFlowNode> CFGNodes = new HashSet<ControlFlowNode>();
    int nodeID;
    ControlFlowNode buggyCFGNode;
    HashMap<ControlFlowNode, Integer> nodeIds;
    static private int maxLength = 0;

    public intervalNode(ControlFlowNode node) {
        nodeID = node.getId();
        CFGNodes.add(node);
    }
    public void addNode(ControlFlowNode node) {
        CFGNodes.add(node);
    }
    public void addNodes(HashSet<ControlFlowNode> nodes) {
        CFGNodes.addAll(nodes);
    }
    public HashSet<ControlFlowNode> getNode() {
        return CFGNodes;
    }
    public ControlFlowNode findhNode() {
        for (ControlFlowNode n : CFGNodes) {
            if (n.getId() == nodeID) {
                return n;
            }
        }
        return null;
    }

    public void updateBuggyCFGNode(ControlFlowNode n) {
        buggyCFGNode = n;
    }

    public ControlFlowNode findNodeByID(int id) {
        for (ControlFlowNode n : CFGNodes) {
            if (n.getId() == id) {
                return n;
            }
        }
        return null;
    }

    private List<ControlFlowNode> next(ControlFlowNode node) {
        ArrayList<ControlFlowNode> result = new ArrayList<ControlFlowNode>();
        for (ControlFlowNode n : node.next()) {
            if (findNodeByID(n.getId()) != null)
                result.add(n);
        }
        return result;
    }

    private List<ControlFlowNode> prev(ControlFlowNode node) {
        ArrayList<ControlFlowNode> result = new ArrayList<ControlFlowNode>();
        for (ControlFlowNode n : node.prev()) {
            if (findNodeByID(n.getId()) != null)
                result.add(n);
        }
        return result;
    }

    private List<ControlFlowEdge> edges() {
        return null;
    }

    public void printNode(int ind) {
        if (ind == 0) {
            ind = nodeID;
        }
        ControlFlowNode n = findNodeByID(ind);
        if (n.getStatement() != null)
            System.out.print(n.getStatement().toString());
        else
            System.out.print("empty statement\n");
    }

    public int getNodeID() {
        return nodeID;
    }

    public void constructIntervalCFG(){
        returnJsonData(0);
        for (ControlFlowNode n : CFGNodes) {
            returnToken(n);
        }
    }

    private void initNodeMap() {
        int i = 0;
        if (nodeIds == null) {
            nodeIds = new HashMap<ControlFlowNode, Integer>();
            for (ControlFlowNode n : CFGNodes) {
                nodeIds.put(n, i);
                ++i;
            }
        }
    }

    private List<int[]> returnGraphRep() {
        // [[0, 1, 1], [1, 1, 2], ...]
        // 0->1 whose edge type is 1, 1->2 whose edge tpye is 2.
        List<int[]> graph = new ArrayList<int[]>();
        initNodeMap();

        for (ControlFlowNode n : CFGNodes) {
            for (ControlFlowNode nn : next(n)) {
                // the edge type should start at 1.
                int[] data = {nodeIds.get(n), 1, nodeIds.get(nn)};
                graph.add(data);
            }
        }
        if (graph.isEmpty()) {
            int[] data = {0, 1, 0};
            graph.add(data);
        }
        return graph;
    }

    private int[][] returnAdjMatrix() {
        int matN = CFGNodes.size();
        int [][] adjMatrix = new int[matN][matN];
        initNodeMap();

        for (ControlFlowNode n : CFGNodes) {
            for (ControlFlowNode nn : next(n)) {
                adjMatrix[nodeIds.get(n)][nodeIds.get(nn)] = 1;
            }
        }
        return adjMatrix;
    }

    public static tokenVisitor returnToken(ControlFlowNode n) {
        CtElement ctele = n.getStatement();
        if (ctele == null) {
            tokenVisitor visitor = new tokenVisitor();
            if (n.getKind() == BranchKind.BEGIN)
                visitor.updateVector(tokenIndex.CFGBegin, ctele);
            else if (n.getKind() == BranchKind.EXIT)
                visitor.updateVector(tokenIndex.CFGEnd, ctele);
            else
                visitor.updateVector(tokenIndex.Other, ctele);
            return visitor;
        }
        tokenVisitor visitor = new tokenVisitor();
        visitor.scan(ctele);

        /*

        int [] tokenSeq = visitor.getTokenSeq();
        int i = tokenSeq.length;
        if (i > maxLength)
        {
            System.out.println(i);
            System.out.println(ctele.toString());
        }
        if (i > 40) {
            tokenVisitor visitor2 = new tokenVisitor();
            visitor2.scan(ctele);
            for ( CtElement cte : ctele.asIterable()) {
                Class c = cte.getClass();
                System.out.println(cte.toString());
            }

        }
        */
        return visitor;

    }

    public JSONObject returnJsonData(int intervalID) {
        JSONObject graphJson = new JSONObject();
        List<int[]> graph = returnGraphRep();
        int[][] node_features = new int[nodeIds.size()][tokenIndex.Size];
        List<int[]> convRep = new ArrayList<>();
        int[] numOfFeatures = new int[nodeIds.entrySet().size()];
        int i = 0;

        for (Map.Entry<ControlFlowNode, Integer> item : nodeIds.entrySet()) {
            ControlFlowNode node = item.getKey();
            Integer id = item.getValue();
            tokenVisitor vis = returnToken(node);
            node_features[id] = vis.getVector();
            int [] tokenSeq = vis.getTokenSeq();
            numOfFeatures[i++] = tokenSeq.length;
            convRep.add(tokenSeq);
        }

        /*
        JSONArray a1 = new JSONArray();
        JSONArray a2 = new JSONArray();
        a1.add(0);
        a2.add(a1);
        graphJson.put("targets", a2);
        */

        graphJson.put("insideinterval", 1);
        graphJson.put("intervalID", intervalID);
        graphJson.put("graph", commom.ArrayToList(graph));
        graphJson.put("node_features", commom.ArrayToList(node_features));
        graphJson.put("convRep", commom.ArrayToList(convRep));
        graphJson.put("numOfFeatures", commom.ArrayToList(numOfFeatures));
        graphJson.put("bugPos", commom.returnBuggyNode(nodeIds, buggyCFGNode));

        return graphJson;
    }
}
