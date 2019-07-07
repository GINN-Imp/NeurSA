import fr.inria.controlflow.*;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.*; 
import spoon.reflect.code.CtReturn;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.declaration.CtMethod;

import java.util.*;

import org.apache.commons.collections4.ListUtils;

import javax.naming.ldap.Control;


public class CFGInt {
    List<ControlFlowGraph> graphs = new ArrayList<>();
    List<IntervalDerivedGraph> ISGs = new ArrayList<IntervalDerivedGraph>();
    List<ProgGraph> progGraphs = new ArrayList<ProgGraph>();
    boolean ifPrint = false;
    int cleanOutputCount = -1;
    public CFGInt() {
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

    private IntervalDerivedGraph retIntervals(CtMethod corBM, CtMethod element) {
        ControlFlowGraph graph = cfgConstruct(element);
        graphs.add(graph);
        IntervalDerivedGraph intervals = constructIntervals(graph, element);
        intervals.addCorBuggyMethod(corBM);
        return intervals;
    }

    public void addGraphs(CtMethod element, Map<CtMethod, BuggyInfo> buggyInfoMap, boolean isFix,
                          boolean outputClean, boolean outputCleanAll) {
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
            IntervalDerivedGraph intervals = retIntervals(corBM, element);
            if (!isFix)
                intervals.addBuggyLine(buggyLine);
            ISGs.add(intervals);
        }
        else {
            assert(cleanOutputCount != -1);
            if ((outputClean && cleanOutputCount != 0) || outputCleanAll) {
                IntervalDerivedGraph intervals = retIntervals(corBM, element);
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
        IntervalDerivedGraph ISG = new IntervalDerivedGraph();
        ISG.addEntryMethod(method);

        List<ControlFlowNode> H = ISG.ReturnEntryNodes(graph);
        if (H == null)
            return null;
        HashSet<ControlFlowNode> processed = new HashSet<ControlFlowNode>();
        while(!H.isEmpty()) {
            ControlFlowNode h = H.remove(0);
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

    public void dumpJson(String outputXML, boolean isDumpCFG, boolean isDumpAST) {
        List<JSONObject> intervalLists = new ArrayList<JSONObject>();
        for (IntervalDerivedGraph ISG : ISGs) {
            intervalLists.add(ISG.returnJsonData());
        }
        String outputJson = outputXML;
        writeJsonFile(intervalLists, outputJson);

        if (isDumpCFG){
            intervalLists.clear();
            outputJson = outputJson.split("\\.(?=[^\\.]+$)")[0];

            outputJson += "-CFG.json";
            for (IntervalDerivedGraph ISG : ISGs) {
                intervalLists.add(ISG.returnCFGJsonData());
            }
            writeJsonFile(intervalLists, outputJson);
            outputJson = outputXML;
        }
        if (isDumpAST) {
            intervalLists.clear();
            outputJson = outputJson.split("\\.(?=[^\\.]+$)")[0];
            outputJson += "-AST.json";
            for (IntervalDerivedGraph ISG : ISGs) {
                intervalLists.add(ISG.returnASTJsonData());
            }
            writeJsonFile(intervalLists, outputJson);
            outputJson = outputXML;
        }
    }

    private void writeJsonFile(List<JSONObject> jsonFile, String outputFileName) {
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
