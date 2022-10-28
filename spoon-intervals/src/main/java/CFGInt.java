import fr.inria.controlflow.*;

import java.io.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*;
import spoon.reflect.code.CtInvocation;
import spoon.reflect.code.CtReturn;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtExecutable;
import spoon.reflect.declaration.CtMethod;

import java.util.*;

import org.apache.commons.collections4.ListUtils;
import spoon.reflect.CtModel;
import spoon.reflect.reference.CtExecutableReference;
import spoon.reflect.visitor.Filter;

import javax.naming.ldap.Control;


public class CFGInt {
    List<ControlFlowGraph> graphs = new ArrayList<>();
    List<IntervalDerivedGraph> ISGs = new ArrayList<IntervalDerivedGraph>();
    List<ProgGraph> progGraphs = new ArrayList<ProgGraph>();
    boolean ifPrint = false;
    boolean useDFG = false;
    int cleanOutputCount = -1;
    CtModel model;
    public CFGInt(CtModel m) {
        model = m;
    }

    private CtMethod returnCorBuggyMethod(Map<CtMethod, BuggyInfo> buggyInfoMap, CtMethod method) {
        SourcePosition position = method.getPosition();
		if(position.isValidPosition()) {
            for (Map.Entry<CtMethod, BuggyInfo> bi : buggyInfoMap.entrySet()) {
                if (position.getFile().toString().endsWith(bi.getValue().targetFile)) {
                    return bi.getKey();
                }
            }
        }
        return null;
    }

    private HashSet<ControlFlowNode> retCFGNodes(ControlFlowGraph g) {
        HashSet<ControlFlowNode> tmp = new HashSet<>();
        for (ControlFlowNode n : g.vertexSet()) {
            // we don't need to predict nodes without statements.
            CtElement ele = n.getStatement();
            if (ele == null)
                continue;
            tmp.add(n);
        }
        return tmp;
    }

    private IntervalDerivedGraph retIntervals(CtMethod corBM, CtMethod element, int inlineLevel) {
        ControlFlowGraph graph = cfgConstruct(element);
        HashSet<ControlFlowNode> CFGNodesWithoutCallee = retCFGNodes(graph);

        List<CtMethod> callees = analyzeCallees(graph, inlineLevel);
        graphs.add(graph);
        IntervalDerivedGraph intervals = constructIntervals(graph, element);
        intervals.addCFGNodesWithoutCallee(CFGNodesWithoutCallee);
        intervals.addCorBuggyMethod(corBM);
        intervals.addCallees(callees);
        return intervals;
    }

    List<CtMethod> analyzeCallees(ControlFlowGraph bm, int inlineLevel) {
        List<CtMethod> callees = new ArrayList<>();
        List<ControlFlowNode> corNodes = new ArrayList<>();
        if (inlineLevel == 0)
            return callees;

        for ( ControlFlowNode node : bm.vertexSet()) {
            CtElement cte = node.getStatement();
            if (cte == null)
                continue;
            for ( CtElement ele : cte.asIterable()) {
                if (ele instanceof  CtExecutableReference) {

                    CtExecutableReference CI = (CtExecutableReference) ele;
                    CtExecutable t = CI.getDeclaration();
                    if (t instanceof  CtMethod) {
                        CtMethod method = (CtMethod) t;
                        if (method.getBody() == null)
                            continue;;
                        callees.add(method);
                        // it is ok since list.add adds a reference.
                        corNodes.add(node);
                    }
                }
            }
        }

        List<CtMethod> subCallees = new ArrayList<>();
        for ( int index = 0; index < callees.size(); index++) {
            CtMethod method = callees.get(index);
            ControlFlowNode node = corNodes.get(index);
            ControlFlowGraph graph = cfgConstruct(method);
            subCallees.addAll(analyzeCallees(graph, inlineLevel - 1));
            mergeTwoGraphs(bm, graph);
            ControlFlowNode calleeExit = graph.getExitNode();
            List<ControlFlowNode> calleeBegins = graph.findNodesOfKind(BranchKind.BEGIN);
            if (calleeBegins.size() != 1)
                System.out.println("ERROR: Callee Begin is larger than 1.");
            ControlFlowNode calleeBegin = calleeBegins.get(0);
            // Modify the kind since it is no longer a BEGIN node.
            calleeBegin.setKind(BranchKind.STATEMENT);
            calleeExit.setKind(BranchKind.STATEMENT);
            for (ControlFlowNode next : node.next()) {
                bm.removeEdge(node, next);
                bm.addEdge(node, calleeBegin);
                bm.addEdge(calleeExit, next);
            }
        }
        callees.addAll(subCallees);
        return callees;
    }

    public  void mergeTwoGraphs(ControlFlowGraph g1, ControlFlowGraph g2) {
        for (ControlFlowNode n : g2.vertexSet()) {
            g1.addVertex(n);
        }
        for (ControlFlowEdge e : g2.edgeSet()) {
            g1.addEdge(e.getSourceNode(), e.getTargetNode());
        }

    }

    public void addGraphs(CtMethod element, Map<CtMethod, BuggyInfo> buggyInfoMap, boolean isFix,
                          boolean outputClean, boolean outputCleanAll, int inlineLevel, boolean useDFG) {
        this.useDFG = useDFG;
        CtMethod corBM = returnCorBuggyMethod(buggyInfoMap, element);
        if (corBM == null)
            return;

        if (cleanOutputCount == -1) {
            cleanOutputCount = buggyInfoMap.size();
        }
        int buggyLine = 0;
        if (buggyInfoMap == null)
            buggyLine = -1;
        else {
            BuggyInfo b = buggyInfoMap.get(element);
            if (b != null)
                buggyLine = b.targetLine;
        }

        if (buggyLine != 0) {
            IntervalDerivedGraph intervals = retIntervals(corBM, element, inlineLevel);
            if (!isFix)
                intervals.addBuggyLine(buggyLine);
            ISGs.add(intervals);
        }
        else {
            assert(cleanOutputCount != -1);
            if ((outputClean && cleanOutputCount != 0) || outputCleanAll) {
                IntervalDerivedGraph intervals = retIntervals(corBM, element, inlineLevel);
                ISGs.add(intervals);
                cleanOutputCount--;
            }
        }

    }


    public ControlFlowGraph cfgConstruct(CtMethod element){
        ControlFlowBuilder builder = new ControlFlowBuilder();
        ControlFlowGraph graph = builder.build(element);
        graph.simplify();
        if(ifPrint) {
            System.out.println(element.getSimpleName()+"\n");
            System.out.println(graph.toGraphVisText());
        }

        return graph;
    }

    private HashSet<ControlFlowNode> analyzeINodes(intervalNode I) {
        HashSet<ControlFlowNode> nodes = I.getNode();
        HashSet<ControlFlowNode> sucessors = new HashSet<ControlFlowNode>();
        HashSet<ControlFlowNode> setOfn = new HashSet<ControlFlowNode>();
        for (ControlFlowNode node : nodes) {
            HashSet<ControlFlowNode> succs = new HashSet<ControlFlowNode>(node.next());
            sucessors.addAll(succs);
        }
        //sucessors  = ListUtils.subtract(sucessors, nodes);
        sucessors.removeAll(nodes);
        for (ControlFlowNode node : sucessors) {
            HashSet<ControlFlowNode> preds = new HashSet<ControlFlowNode>(node.prev());
            if(nodes.containsAll(preds)) {
                setOfn.add(node);
            }
        }
        return setOfn;
    }

    private HashSet<ControlFlowNode> findNextHeaders(intervalNode I, ControlFlowGraph graph,
                                                     HashSet<ControlFlowNode> processed) {
        HashSet<ControlFlowNode> headers = new HashSet<ControlFlowNode>();
        HashSet<ControlFlowNode> INodes = I.getNode();

        for (ControlFlowNode node : graph.vertexSet()) {
            if (INodes.contains(node) || processed.contains(node))
                continue;
            HashSet<ControlFlowNode> preds = new HashSet<ControlFlowNode>(node.prev());
            for (ControlFlowNode m : preds) {
                if (INodes.contains(m)) {
                    headers.add(node);
                    break;
                }
            }

        }
        return headers;
    }

    public IntervalDerivedGraph constructIntervals(ControlFlowGraph graph, CtMethod method){
        IntervalDerivedGraph ISG = new IntervalDerivedGraph(useDFG);
        ISG.addEntryMethod(method);

        List<ControlFlowNode> entryNodes = ISG.ReturnEntryNodes(graph);
        HashSet<ControlFlowNode> H = new HashSet<>(entryNodes);
        if (H == null)
            return null;
        HashSet<ControlFlowNode> processed = new HashSet<ControlFlowNode>();
        while(!H.isEmpty()) {
            ControlFlowNode h = H.iterator().next();
            H.remove(h);
            intervalNode I = new intervalNode(h);
            HashSet<ControlFlowNode> setOfn;
            do {
                setOfn = analyzeINodes(I);
                I.addNodes(setOfn);
            } while (!setOfn.isEmpty());
            processed.addAll(I.getNode());
            H.addAll(findNextHeaders(I, graph, processed));
            //I.constructIntervalCFG();
            ISG.addVertex(I);
        }
        ISG.linkEdges();
        //ISG.returnAdjMatrix();
        if(ifPrint)
            ISG.print();
        return ISG;
    }

    public void dumpJson(String outputXML, boolean isDumpCFG, boolean isDumpAST, String projName) {
        List<JSONObject> intervalLists = new ArrayList<JSONObject>();
        for (IntervalDerivedGraph ISG : ISGs) {
            intervalLists.add(ISG.returnJsonData());
        }
        String outputJson = outputXML;
        writeJsonFileByText(intervalLists, outputJson, projName);

        if (isDumpCFG){
            intervalLists.clear();
            outputJson = outputJson.split("\\.(?=[^\\.]+$)")[0];

            outputJson += "-CFG.json";
            for (IntervalDerivedGraph ISG : ISGs) {
                intervalLists.add(ISG.returnCFGJsonData());
            }
            writeJsonFileByText(intervalLists, outputJson, projName);
            outputJson = outputXML;
        }
        if (isDumpAST) {
            intervalLists.clear();
            outputJson = outputJson.split("\\.(?=[^\\.]+$)")[0];
            outputJson += "-AST.json";
            for (IntervalDerivedGraph ISG : ISGs) {
                intervalLists.add(ISG.returnASTJsonData());
            }
            writeJsonFileByText(intervalLists, outputJson, projName);
            outputJson = outputXML;
        }
    }

    private void writeJsonFileByText(List<JSONObject> jsonFile, String outputFileName, String projName) {
        File outputFile = new File(outputFileName);
        StringBuilder contentBuilder = new StringBuilder();
        boolean addComma = true;
        String data = null;
        if (outputFile.exists()) {

            try (BufferedReader br = new BufferedReader(new FileReader(outputFileName)))
            {

                String sCurrentLine;
                while ((sCurrentLine = br.readLine()) != null)
                {
                    contentBuilder.append(sCurrentLine).append("\n");
                }
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
            data = contentBuilder.toString();
            if (data.length() < 4 ) {
                data = "[";
                addComma = false;
            }
            else
                data = data.substring(0, data.length()-2);
        }
        else {
            data = "[";
            addComma = false;
        }


        for (JSONObject array : jsonFile) {
            array.put("projName", projName);
            if (addComma) {
                data += ",";
            }
            else
                addComma = true;
            data += array.toJSONString();
        }
        data += "]";

        try (FileWriter file = new FileWriter(outputFile)) {
            file.write(data);
            file.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void writeJsonFile(List<JSONObject> jsonFile, String outputFileName, String projName) {
        JSONArray intervalLists = null;
        File outputFile = new File(outputFileName);
        if (outputFile.exists()) {
            try (FileReader f = new FileReader(outputFile)) {
                JSONParser parser = new JSONParser();
                intervalLists = (JSONArray) parser.parse(f);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        else {
            intervalLists = new JSONArray();
        }

        for (JSONObject array : jsonFile) {
            array.put("projName", projName);
            intervalLists.add(array);
        }

        try (FileWriter file = new FileWriter(outputFile)) {
            file.write(intervalLists.toJSONString());
            file.flush();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
