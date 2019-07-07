import com.github.gumtreediff.tree.ITree;
import fr.inria.controlflow.BranchKind;
import fr.inria.controlflow.ControlFlowEdge;
import fr.inria.controlflow.ControlFlowGraph;
import fr.inria.controlflow.ControlFlowNode;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.declaration.CtElement;

import java.util.*;

public class IntervalDerivedGraph extends DefaultDirectedGraph<intervalNode, intervalEdge>{

    HashMap<intervalNode, Integer> nodeIds;
    CtMethod corBuggyMethod = null;
    intervalNode buggyNode = null;
    CtMethod baseMethod = null;
    int buggyLine = 0;
    ControlFlowNode buggyCFGNode = null;
    ControlFlowGraph CFG = null;
    HashMap<ControlFlowNode, Integer> CFGNodeIds = null;

    public IntervalDerivedGraph(Class<? extends intervalEdge> edgeClass) {
        super(edgeClass);
    }

    public IntervalDerivedGraph() {
        super(intervalEdge.class);
    }

    List<ControlFlowNode> ReturnEntryNodes(ControlFlowGraph graph) {
        CFG = graph;
        return graph.findNodesOfKind(BranchKind.BEGIN);
    }

    public void addEntryMethod(CtMethod method) {
        baseMethod = method;
    }

    public void addCorBuggyMethod(CtMethod method) { corBuggyMethod = method;}

    @Override
    public intervalEdge addEdge(intervalNode source, intervalNode target) {
        if (!containsVertex(source)) {
            addVertex(source);
        }
        if (!containsVertex(target)) {
            addVertex(target);
        }
        return super.addEdge(source, target);
    }
    public intervalNode findNodeById(int id) {
        for (intervalNode n : vertexSet()) {
            if (n.findNodeByID(id) != null) {
                return n;
            }
        }
        return null;
    }
    public void linkEdges(){
        // link edges between intervalNodes according
        // to the control flow graph
        for (intervalNode n : vertexSet()) {
            for (ControlFlowNode h : n.getNode()) {
                for (ControlFlowNode e : h.next()) {
                    intervalNode next = findNodeById(e.getId());
                    if (next.getNodeID() != n.getNodeID()) {
                        addEdge(n, next);
                    }
                }
            }
        }
        initNode();
    }
    public void print(){
        StringBuilder sb = new StringBuilder("digraph ").append("null").append(" { \n");
        //sb.append("exit [shape=doublecircle];\n");
        sb.append("node [fontsize = 8];\n");


        int i = 0;
        for (intervalNode n : vertexSet()) {
            sb.append(n.getNodeID() + " [label=" + n.getNodeID() + "];\n");
        }

        for (intervalEdge e : edgeSet()) {
            sb.append(e.getSourceNode().getNodeID() + " -> " +
                    e.getTargetNode().getNodeID() + " ;\n");
        }

        sb.append("}\n");
        System.out.print(sb.toString());
    }

    public void addBuggyLine(int buggyLine) {
        int closest=0;
        ControlFlowNode tmph = null;
        intervalNode tmpn = null;
        for (intervalNode n : vertexSet()) {
            for (ControlFlowNode h : n.getNode()) {
                CtElement stmt = h.getStatement();
                if (stmt == null) {
                    continue;
                }
                SourcePosition source = stmt.getPosition();
                if (source.isValidPosition()) {
                    int curLine = source.getLine();
                    if (curLine == buggyLine) {
                        buggyNode = n;
                        buggyCFGNode = h;
                        this.buggyLine = buggyLine;
                        break;
                    }
                    else if (curLine > closest && curLine < buggyLine) {
                        closest = curLine;
                        tmph = h;
                        tmpn = n;
                    }
                }
            }
        }

        assert buggyNode != null : "BuggyLine must be found!";
        if (buggyNode != null) {
            // for line prediction, we should avoid confuse lines.
            // therefore, we should filter instances whose bugpos is all 0.
            buggyNode.updateBuggyCFGNode(buggyCFGNode);
        }

        if (buggyNode == null){
            if (tmpn != null) {
                buggyNode = tmpn;
                buggyCFGNode = tmph;
                System.out.println("using tmp node.");
                buggyNode.updateBuggyCFGNode(buggyCFGNode);
            }
            else {
                //TODO: ohhhhh myyyyyyy goooooood
                System.out.println("Still not found.");
            }
        }

    }

    private void initNode() {
        if (nodeIds == null) {
            int i = 0;
            nodeIds = new HashMap<intervalNode, Integer>();
            for (intervalNode n : vertexSet()) {
                nodeIds.put(n, i);
                ++i;
            }
        }
    }


    private List<int[]> returnGraphRep() {
        // [[0, 1, 1], [1, 1, 2], ...]
        // 0->1 whose edge type is 1, 1->2 whose edge tpye is 2.
        List<int[]> graph = new ArrayList<int[]>();

        for (intervalEdge e : edgeSet()) {
            // the edge type should start at 1.
            int[] data = {nodeIds.get(e.getSourceNode()), 1, nodeIds.get(e.getTargetNode())};
            graph.add(data);
        }
        if (graph.isEmpty()) {
            int[] data = {0, 1, 0};
            graph.add(data);
        }

        return graph;
    }

    public boolean[][] returnAdjMatrix() {
        int matN = vertexSet().size();
        boolean[][] adjMatrix = new boolean[matN][matN];
        for (intervalEdge e : edgeSet()) {
            adjMatrix[nodeIds.get(e.getSourceNode())][nodeIds.get(e.getTargetNode())] = true;
        }
        return adjMatrix;
    }

    public JSONObject returnJsonData() {
        JSONObject graphJson = new JSONObject();
        List<int[]> graph = returnGraphRep();
        for (intervalNode n : vertexSet())
            graphJson.put(nodeIds.get(n), n.returnJsonData(nodeIds.get(n)));

        JSONArray a2 = commom.returnTarget(nodeIds, buggyNode);
        JSONArray bugPos = commom.returnBuggyNode(nodeIds, buggyNode);

        graphJson.put("insideinterval", 0);
        graphJson.put("targets", a2);
        graphJson.put("numOfNode", vertexSet().size());
        graphJson.put("graph", commom.ArrayToList(graph));
        graphJson.put("bugPos", bugPos);
        graphJson.put("fileHash", commom.returnFileInd(corBuggyMethod));

        return graphJson;
    }


    public JSONObject returnASTJsonData() {
        JSONObject graphJson = new JSONObject();

        ProgGraph astBasedProgGraph = new ProgGraph(baseMethod, buggyCFGNode, CFGNodeIds);
        List<int[]> graph = astBasedProgGraph.graph;


        int[][] node_features = astBasedProgGraph.returnFeatures();
        boolean[] node_mask = astBasedProgGraph.returnNodeMask();

        JSONArray a2 = commom.returnTarget(CFGNodeIds, buggyCFGNode);
        JSONArray bugPos = astBasedProgGraph.returnBugPos();

        // The bug pos of AST is the same to CFG, since building CFG is based on AST.
        graphJson.put("targets", a2);
        graphJson.put("node_features",commom.ArrayToList(node_features) );
        graphJson.put("node_mask",commom.ArrayToList(node_mask) );
        graphJson.put("graph", commom.ArrayToList(graph));
        graphJson.put("bugPos", bugPos);
        graphJson.put("fileHash", commom.returnFileInd(corBuggyMethod));
        return graphJson;
    }

    private void initCFGNodeID() {
        if (CFGNodeIds != null)
            return;
        CFGNodeIds = new HashMap<ControlFlowNode, Integer>();
        int i = 0;
        for (ControlFlowNode n : CFG.vertexSet()) {
            CFGNodeIds.put(n, i);
            ++i;
        }
    }

    public JSONObject returnCFGJsonData() {
        JSONObject graphJson = new JSONObject();
        List<int[]> graph = new ArrayList<int[]>();

        initCFGNodeID();

        for (ControlFlowEdge e : CFG.edgeSet()) {
            // the edge type should start at 1.
            int[] data = {CFGNodeIds.get(e.getSourceNode()), 1, CFGNodeIds.get(e.getTargetNode())};
            graph.add(data);
        }
        if (CFG.vertexSet().isEmpty()) {
            int[] data = {0, 1, 0};
            graph.add(data);
        }

        int[][] node_features = new int[CFGNodeIds.size()][tokenIndex.Size];
        boolean[] node_mask = new boolean[CFGNodeIds.size()];
        Arrays.fill(node_mask, true);


        for (Map.Entry<ControlFlowNode, Integer> item : CFGNodeIds.entrySet()) {
            ControlFlowNode node = item.getKey();
            Integer id = item.getValue();
            tokenVisitor vis = intervalNode.returnToken(node);
            node_features[id] = vis.getVector();
        }

        JSONArray a2 = commom.returnTarget(CFGNodeIds, buggyCFGNode);
        JSONArray bugPos = commom.returnBuggyNode(CFGNodeIds, buggyCFGNode);

        graphJson.put("targets", a2);
        graphJson.put("node_mask",commom.ArrayToList(node_mask) );
        graphJson.put("node_features",commom.ArrayToList(node_features) );
        graphJson.put("graph", commom.ArrayToList(graph));
        graphJson.put("bugPos", bugPos);
        graphJson.put("fileHash", commom.returnFileInd(corBuggyMethod));

        return graphJson;
    }
}
