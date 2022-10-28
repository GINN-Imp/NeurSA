import com.github.gumtreediff.tree.ITree;
import fr.inria.controlflow.ControlFlowNode;
import gumtree.spoon.builder.SpoonGumTreeBuilder;
import org.json.simple.JSONArray;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtMethod;

import java.util.*;

public class ProgGraph {
    CtMethod method = null;
    ITree iTree = null;
    ITree buggyNode = null;
    HashMap<ITree, Integer> nodeIds = new HashMap<ITree, Integer>();
    HashMap<CtElement, List<ITree>> eleTreeMap = new HashMap<>();
    HashSet<ITree> targetNodes = new HashSet<>();
    int treeNodeCount = 0;
    List<ITree> leafNode = new ArrayList<ITree>();
    List<int[]> graph = new ArrayList<int[]>();
    boolean[] node_mask = null;
    HashSet<ControlFlowNode> CFGNodesWithoutInline = null;

    public ProgGraph(CtMethod method, ControlFlowNode buggyCFGNode,
                     HashMap<ControlFlowNode, Integer> CFGNodeIds, List<CtMethod> callees,
                     HashSet<ControlFlowNode> CFGNodesWithoutInline) {
        this.method = method;
        this.CFGNodesWithoutInline = CFGNodesWithoutInline;
        SpoonGumTreeBuilder scanner = new SpoonGumTreeBuilder();
        iTree = scanner.getTree(method);
        retProg();
        for (CtMethod callee : callees) {
            iTree = scanner.getTree(callee);
            retProg();
        }
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
            CtElement ctele = common.returnElementOfITree(node);
            if (!eleTreeMap.containsKey(ctele))
                eleTreeMap.put(ctele, new ArrayList<ITree>());
            eleTreeMap.get(ctele).add(node);
        }
    }

    private void locateBuggyNode(CtElement ele) {
        if (eleTreeMap.containsKey(ele)) {
            List<ITree> tmp = eleTreeMap.get(ele);
            buggyNode = tmp.get(0);
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

            CtElement ctele = common.returnElementOfITree(node);
            tokenVisitor visitor = new tokenVisitor();
            visitor.scan(ctele);
            node_features[id] = visitor.getVector();
            if (targetNodes.contains(node)) {
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
        JSONArray bugPos = common.returnBuggyNode(nodeIds, buggyNode, node_mask);
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
            ControlFlowNode n = item.getKey();
            CtElement ele = n.getStatement();
            if (ele == null)
                continue;
            if (eleTreeMap.containsKey(ele)) {
                treeNodeCount++;
                if (CFGNodesWithoutInline.contains(n)) {
                    targetNodes.addAll(eleTreeMap.get(ele));
                }
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
