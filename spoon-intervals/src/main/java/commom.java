import com.github.gumtreediff.tree.ITree;
import org.json.simple.JSONArray;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtMethod;

import java.util.HashMap;
import java.util.List;
import java.util.Arrays;

public class commom {
    public static JSONArray ArrayToList(List<int[]> graph) {
        JSONArray a2 = new JSONArray();
        for (int [] vec : graph) {
            JSONArray a1 = ArrayToList(vec);
            a2.add(a1);
        }

        return a2;
    }

    public static JSONArray ListToJsonArray(List<Integer> graph) {
        JSONArray a2 = new JSONArray();
        for (Integer vec : graph) {
            a2.add(vec);
        }
        return a2;
    }

    public static JSONArray ArrayToList(int[][] graph) {
        JSONArray a2 = new JSONArray();
        for (int [] vec : graph) {
            JSONArray a1 = ArrayToList(vec);
            a2.add(a1);
        }

        return a2;
    }

    public static JSONArray ArrayToList(int[] graph) {
        JSONArray a1 = new JSONArray();
        for (int i : graph)
            a1.add(i);

        return a1;
    }

    public static JSONArray ArrayToList(boolean[] graph) {
        JSONArray a1 = new JSONArray();
        for (boolean i : graph)
            a1.add(i);

        return a1;
    }

    public static float computeTarget(int nodeID, int totalNum) {
        //float res = (float) (nodeID + 1) / (totalNum + 1);
        float res = (float)1.0;
        return res;
    }



    public static <T> JSONArray returnTarget(HashMap<T, Integer> nodeIds, T buggyNode) {
        JSONArray a1 = new JSONArray();
        JSONArray a2 = new JSONArray();
        float res = 0;
        if (buggyNode != null) {
            //res = (float) (nodeIds.get(buggyNode) + 0.0001) / nodeIds.size();
            res = commom.computeTarget(nodeIds.get(buggyNode), nodeIds.size());
        }
        a1.add(res);
        a2.add(a1);
        return a2;
    }


    public static JSONArray returnFileInd(CtMethod method) {
        JSONArray a1 = new JSONArray();
        int res = 0;
        //String res = null;
        if (method != null) {
            //res = (float) (nodeIds.get(buggyNode) + 0.0001) / nodeIds.size();
            res = method.hashCode();
            //res = method.getPosition().toString();
        }
        a1.add(res);
        return a1;
    }

    public static CtElement returnElementOfITree(ITree iTree) {
        return (CtElement)iTree.getMetadata("spoon_object");
    }

    public static <T> JSONArray returnBuggyNode(HashMap<T, Integer> nodeIds, T buggyNode) {
        int[] buggyNodePos = new int[nodeIds.size()];
        if (buggyNode != null) {
            buggyNodePos[nodeIds.get(buggyNode)] = 1;
        }
        JSONArray a = commom.ArrayToList(buggyNodePos);
        return a;
    }
}

