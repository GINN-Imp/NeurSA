import org.jgrapht.graph.DefaultEdge;

public class intervalEdge extends DefaultEdge {
    public intervalNode getTargetNode() {
        return (intervalNode) getTarget();
    }

    public intervalNode getSourceNode() {
        return (intervalNode) getSource();
    }
}

