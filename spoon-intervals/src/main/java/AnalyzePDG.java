import com.github.gumtreediff.tree.ITree;
import gumtree.spoon.builder.SpoonGumTreeBuilder;
import org.apache.commons.cli.*;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.code.*;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.factory.Factory;
import spoon.reflect.visitor.Filter;
import spoon.reflect.visitor.filter.TypeFilter;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ghaffarian.progex.java.JavaPDGBuilder;
import ghaffarian.progex.graphs.pdg.ProgramDependeceGraph;
import spoon.support.compiler.VirtualFile;


public class AnalyzePDG {
    CtMethod curMethod = null;
    Map<Object, Integer> nodeIDMap = new HashMap<>();
    int lastNodeID = 0;
    List<int[]> graph = new ArrayList<int[]>();
    Map<Integer, int[]> featureMap = new HashMap<>();

    private int returnID(Object o) {
        if (!nodeIDMap.containsKey(o))
            nodeIDMap.put(o, lastNodeID++);
        return nodeIDMap.get(o);
    }

    private CtElement findEleInMethod(CtElement method, String label) {
        for (CtElement stmt : method.getElements(new TypeFilter<>(CtElement.class))){
            if (stmt.toString().equals(label)) {
                return stmt;
            }
        }
        return null;
    }

    private void parseCode(String label, int nodeID) {
        /*
        final Launcher launcher = new Launcher();
        launcher.addInputResource(new VirtualFile(label));
        launcher.getEnvironment().setNoClasspath(true);
        launcher.getEnvironment().setAutoImports(true);
        CtModel model = launcher.buildModel();
        List<CtElement> r = model.getElements(new TypeFilter<>(CtElement.class));
         */
        CtElement tmpStmt = findEleInMethod(curMethod, label);
        if (tmpStmt == null)
            return;

        SpoonGumTreeBuilder scanner = new SpoonGumTreeBuilder();
        ITree iTree = scanner.getTree(tmpStmt);
        // add ast of a statement and node to ast.
        addAST(iTree, graph, 1);
        int[] data = {nodeID, 1, returnID(iTree)};
        graph.add(data);
        return;
    }

    private void addAST(ITree root, List<int[]> graph, int nodeType) {
        int astNodeID = returnID(root);

        CtElement ctele = common.returnElementOfITree(root);
        tokenVisitor visitor = new tokenVisitor();
        visitor.scan(ctele);
        //visitor.getTokenSeq();
        featureMap.put(astNodeID, visitor.getVector());

        for ( ITree child : root.getChildren()) {
            addAST(child, graph, nodeType);
            int[] data = {astNodeID, nodeType, returnID(child)};
            graph.add(data);
        }
    }


    private void transformGraph(String filepath) {
        JSONParser jsonParser = new JSONParser();

        try (FileReader reader = new FileReader(filepath))
        {
            //Read JSON file
            Object obj = jsonParser.parse(reader);

            JSONObject graphInfo = (JSONObject) obj;
            JSONArray edges = (JSONArray) graphInfo.get("edges");
            JSONArray nodes = (JSONArray) graphInfo.get("nodes");
            lastNodeID = nodes.size()+1;
            for (Object n : nodes) {
                JSONObject node = (JSONObject) n;
                String label = node.get("label").toString();
                int nodeID = Integer.parseInt(node.get("id").toString());
                System.out.println(label);
                parseCode(label, nodeID);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void transformPDG(ProgramDependeceGraph pdg, BuggyInfo buggyInfo) throws IOException {
        String outDir = "tempPDGDir";
        File f = new File(buggyInfo.targetFile);
        String filename = f.getName().substring(0, f.getName().indexOf('.'));
        String ctrlPath = outDir + "/"+filename + "-PDG-CTRL.json";
        pdg.CDS.exportJSON("tempPDGDir");
        transformGraph(ctrlPath);
        ctrlPath = outDir + "/"+filename + "-PDG-DATA.json";
        pdg.DDS.exportJSON("tempPDGDir");
        transformGraph(ctrlPath);
    }

    public void analyzeMethods(Map<CtMethod, BuggyInfo> buggyInfoMap) {
        String[] filePaths = new String[1];
        for (Map.Entry<CtMethod, BuggyInfo> bi : buggyInfoMap.entrySet()) {
            try {
                filePaths[0] = bi.getValue().targetFile;
                curMethod = bi.getKey();

                ProgramDependeceGraph[] d = JavaPDGBuilder.buildForAll(filePaths);
                transformPDG(d[0], bi.getValue());
            }
            catch (IOException e){
                System.out.println("File Not Found in AnalyzePDG.");
                System.exit(-1);
            }
        }
    }
}
