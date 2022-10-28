import fr.inria.controlflow.ControlFlowNode;
import fr.inria.dataflow.InitializedVariables;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.reference.CtVariableReference;

import java.util.HashMap;
import java.util.HashSet;

public class DataFlowGraph {
    IntervalDerivedGraph IDG;
    HashMap<ControlFlowNode, ControlFlowNode> VRNodeMap;
    HashMap<CtVariableReference, ControlFlowNode> VRMap;
    HashSet<ControlFlowNode> CFGNodes;
    public HashMap<ControlFlowNode, ControlFlowNode> retVRNodeMap() {return VRNodeMap;}

    public DataFlowGraph(IntervalDerivedGraph d) {
        VRMap = new HashMap<>();
        VRNodeMap = new HashMap<>();
        IDG = d;
        retCFGNodes();
        InitializedVariables vars = new InitializedVariables();

        for (ControlFlowNode n : CFGNodes) {
            vars.run(n);
            for (CtVariableReference v : vars.getInitialized())
                VRMap.put(v, n);
        }
        mapCtVR2CFGNode();
    }
    private void retCFGNodes() {
        CFGNodes = new HashSet<>();

        for (intervalNode IN : IDG.vertexSet()) {
            for (ControlFlowNode n : IN.getNode()) {
                CFGNodes.add(n);
            }
        }
    }

    private void mapCtVR2CFGNode() {
        for (ControlFlowNode n : CFGNodes) {
            CtElement cte = n.getStatement();
            if (cte == null)
                continue;
            for ( CtElement ele : cte.asIterable()) {
                if (ele instanceof  CtVariableReference) {

                    CtVariableReference CV = (CtVariableReference) ele;
                    ControlFlowNode CVNode = VRMap.get(CV);
                    if (CVNode != null) {
                        VRNodeMap.put(CVNode, n);
                    }
                }
            }
        }
    }
}
