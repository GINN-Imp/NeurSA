import com.github.gumtreediff.tree.ITree;
import fr.inria.controlflow.ControlFlowGraph;
import fr.inria.controlflow.ControlFlowNode;
import gumtree.spoon.builder.SpoonGumTreeBuilder;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtMethod;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ProgGraph {
    CtMethod method = null;
    ITree iTree = null;
    ITree buggyNode = null;
    HashMap<ITree, Integer> nodeIds = new HashMap<ITree, Integer>();
    HashMap<CtElement, ITree> eleTreeMap = new HashMap<CtElement, ITree>();
    HashMap<ITree, ControlFlowNode> treeCFGMap = new HashMap<>();
    int treeNodeCount = 0;
    List<ITree> leafNode = new ArrayList<ITree>();
    List<int[]> graph = new ArrayList<int[]>();
    boolean[] node_mask = null;
    public ProgGraph(CtMethod method, ControlFlowNode buggyCFGNode,
                     HashMap<ControlFlowNode, Integer> CFGNodeIds) {
        this.method = method;
        SpoonGumTreeBuilder scanner = new SpoonGumTreeBuilder();
        iTree = scanner.getTree(method);
        retProg();
        initeleTreeMap();
        if (buggyCFGNode != null) {
            CtElement ele = buggyCFGNode.getStatement();
            locateBuggyNode(ele);
        }
        mapToCFG(CFGNodeIds);
    }
    private void initeleTreeMap() {
        for (Map.Entry<ITree, Integer> item : nodeIds.entrySet()) {
            ITree node = item.getKey();
            CtElement ctele = (CtElement)node.getMetadata("spoon_object");
            eleTreeMap.put(ctele, node);
        }
    }

    private void locateBuggyNode(CtElement ele) {
        if (eleTreeMap.containsKey(ele)) {
            buggyNode = eleTreeMap.get(ele);
        }
        if (buggyNode == null) {
            System.out.println("ERROR: node in AST must be found");
        }
    }

    public int[][] returnFeatures() {
        int[][] node_features = new int[nodeIds.size()][tokenIndex.Size];
        node_mask = new boolean[nodeIds.size()];

        for (Map.Entry<ITree, Integer> item : nodeIds.entrySet()) {
            ITree node = item.getKey();
            Integer id = item.getValue();

            CtElement ctele = commom.returnElementOfITree(node);
            tokenVisitor visitor = new tokenVisitor();
            visitor.scan(ctele);
            node_features[id] = visitor.getVector();
            if (treeCFGMap.containsKey(item.getKey())) {
                node_mask[id] = true;
            }
            else {
                node_mask[id] = false;
            }
        }
        return node_features;
    }

    public boolean[] returnNodeMask() {
        return node_mask;
    }

    public JSONArray returnBugPos() {
        int[] oldBuggyNodePos = new int[nodeIds.size()];
        List<Integer> newBuggyNodePos = new ArrayList<>();
        if (buggyNode != null) {
            oldBuggyNodePos[nodeIds.get(buggyNode)] = 1;
        }
        for (int i = 0; i < nodeIds.size(); i++) {
            if (node_mask[i]) {
                newBuggyNodePos.add(oldBuggyNodePos[i]);
            }
        }

        JSONArray bugPos = commom.ListToJsonArray(newBuggyNodePos);
        return bugPos;
    }

    private int getNodeID(ITree o) {
        if (!nodeIds.containsKey(o)) {
            nodeIds.put(o, nodeIds.size());
        }
        return nodeIds.get(o);
    }

    private void addAST(ITree root) {
        int astNodeID = getNodeID(root);
        if (root.getChildren().size() == 0) {
            // each leaf node is sorted.
            leafNode.add(root);
        }
        for ( ITree child : root.getChildren()) {
            addAST(child);
            int[] data = {astNodeID, 1, nodeIds.get(child)};
            graph.add(data);
        }
    }

    public void mapToCFG(HashMap<ControlFlowNode, Integer> CFGNodeIds) {
        //Requires: itree is built.
        for (Map.Entry<ControlFlowNode, Integer> item : CFGNodeIds.entrySet()) {
            CtElement ele = item.getKey().getStatement();
            if (ele == null)
                continue;
            if (eleTreeMap.containsKey(ele)) {
                treeNodeCount++;
                if (treeCFGMap.containsKey(eleTreeMap.get(ele))) {
                    continue;
                }
                treeCFGMap.put(eleTreeMap.get(ele), item.getKey());
            }
            else {
                System.out.println("ERROR in mapToCFG.");
            }
        }


    }

    private void addLeafNodeEdge() {
        int prevLeafNodeID = -1;
        for (ITree leaf : leafNode) {
            int curNodeID = getNodeID(leaf);
            if (prevLeafNodeID != -1) {
                int[] data = {prevLeafNodeID, 2, curNodeID};
                graph.add(data);
            }
            prevLeafNodeID = curNodeID;
        }
    }

    public void retProg(){
        //System.out.println(iTree.toTreeString());
        addAST(iTree);
        addLeafNodeEdge();
        return;
    }

}
