import org.jgrapht.graph.DefaultEdge;

public class intervalEdge extends DefaultEdge {
    int edgeType;
    public intervalEdge() {
        super();
        edgeType = 1;
    }
    public int getEdgeType() {return edgeType;}
    public void setEdgeType(int et) {edgeType = et;}
    public intervalNode getTargetNode() {
        return (intervalNode) getTarget();
    }

    public intervalNode getSourceNode() {
        return (intervalNode) getSource();
    }
}

