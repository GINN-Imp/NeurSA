import org.jgrapht.graph.DefaultDirectedGraph;

public class intervalSetGraph extends DefaultDirectedGraph<intervalSetNode, intervalEdge> {

    public intervalSetGraph(Class<? extends intervalEdge> edgeClass) {
        super(edgeClass);
    }

    public intervalSetGraph() {
        super(intervalEdge.class);
    }
    @Override
    public intervalEdge addEdge(intervalSetNode source, intervalSetNode target) {
        if (!containsVertex(source)) {
            addVertex(source);
        }
        if (!containsVertex(target)) {
            addVertex(target);
        }
        return super.addEdge(source, target);
    }

}
